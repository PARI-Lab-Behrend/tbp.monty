from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


LTPMethod = Literal["default", "ror", "uniform"]


@dataclass(frozen=True)
class LTPResult:
    """Container for Local Ternary Pattern outputs.

    Attributes:
        codes_pos: Positive binary pattern image.
        codes_neg: Negative binary pattern image.
        histogram: Concatenated normalized histogram [pos_hist, neg_hist].
    """

    codes_pos: np.ndarray
    codes_neg: np.ndarray
    histogram: np.ndarray


def _validate_gray_image(gray: np.ndarray) -> np.ndarray:
    """Validate and coerce an input image to a 2D float32 grayscale array."""
    arr = np.asarray(gray)

    if arr.ndim != 2:
        raise ValueError(
            f"`gray` must be a 2D grayscale image, got shape {arr.shape}."
        )

    if arr.size == 0:
        raise ValueError("`gray` must not be empty.")

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"`gray` must be numeric, got dtype {arr.dtype}.")

    return arr.astype(np.float32, copy=False)

def _uniform_lookup(p: int) -> tuple[np.ndarray, int]:
    """Create a lookup table mapping binary codes to uniform-pattern bins.

    A uniform pattern has at most 2 circular bit transitions.
    Non-uniform patterns are mapped to the final bin.

    Returns:
        lut: Array of shape (2**p,) mapping raw codes -> bin index.
        n_bins: Number of bins in the mapped representation.
    """
    n_codes = 1 << p
    lut = np.empty(n_codes, dtype=np.int32)

    uniform_index = 0
    non_uniform_bin = p * (p - 1) + 2  # temporary placeholder

    for code in range(n_codes):
        bits = ((code >> np.arange(p)) & 1).astype(np.uint8)
        transitions = np.count_nonzero(bits != np.roll(bits, -1))

        if transitions <= 2:
            ones = int(bits.sum())
            if ones == 0:
                bin_idx = 0
            elif ones == p:
                bin_idx = p
            else:
                first_one = int(np.argmax(bits))
                rotated = np.roll(bits, -first_one)
                run_length = int(np.argmax(rotated == 0))
                # Unique mapping by (number of ones, starting position)
                bin_idx = 1 + (ones - 1) * p + first_one
            lut[code] = bin_idx
            uniform_index = max(uniform_index, bin_idx)
        else:
            lut[code] = non_uniform_bin

    # Re-pack bins contiguously
    unique_bins = np.unique(lut)
    remap = {old: new for new, old in enumerate(unique_bins)}
    lut = np.array([remap[x] for x in lut], dtype=np.int32)
    n_bins = len(unique_bins)

    return lut, n_bins


def _bilinear_sample(
    image: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Sample image at floating-point coordinates using bilinear interpolation.

    Coordinates outside the image are clipped to the nearest valid boundary.
    """
    h, w = image.shape

    y = np.clip(y, 0.0, h - 1.0)
    x = np.clip(x, 0.0, w - 1.0)

    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    wy = y - y0
    wx = x - x0

    top_left = image[y0, x0]
    top_right = image[y0, x1]
    bottom_left = image[y1, x0]
    bottom_right = image[y1, x1]

    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx

    return top * (1.0 - wy) + bottom * wy


def _compute_ltp_codes(
    gray: np.ndarray,
    p: int,
    r: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw positive and negative LTP codes for each pixel.

    Positive code bit i is 1 if neighbor_i >= center + threshold.
    Negative code bit i is 1 if neighbor_i <= center - threshold.
    """
    if p <= 0:
        raise ValueError(f"`p` must be positive, got {p}.")
    if p > 31:
        raise ValueError(
            "`p` must be <= 31 so codes fit safely in int32/int64 bit operations."
        )
    if r <= 0:
        raise ValueError(f"`r` must be positive, got {r}.")
    if threshold < 0:
        raise ValueError(f"`threshold` must be >= 0, got {threshold}.")

    h, w = gray.shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    center = gray

    codes_pos = np.zeros((h, w), dtype=np.uint32)
    codes_neg = np.zeros((h, w), dtype=np.uint32)

    # Standard circular neighborhood, clockwise, starting at angle 0.
    for i in range(p):
        theta = 2.0 * np.pi * i / p
        dy = -r * np.sin(theta)
        dx = r * np.cos(theta)

        neighbor = _bilinear_sample(gray, yy + dy, xx + dx)

        pos_bit = neighbor >= (center + threshold)
        neg_bit = neighbor <= (center - threshold)

        codes_pos |= pos_bit.astype(np.uint32) << i
        codes_neg |= neg_bit.astype(np.uint32) << i

    return codes_pos, codes_neg

def _ror_lookup(p: int) -> tuple[np.ndarray, int]:
    """Create a lookup table mapping raw codes to compact ROR bins."""
    n_codes = 1 << p
    mask = n_codes - 1
    canonical = np.empty(n_codes, dtype=np.int32)

    for code in range(n_codes):
        x = code
        min_code = code
        for _ in range(1, p):
            x = ((x >> 1) | ((x & 1) << (p - 1))) & mask
            min_code = min(min_code, x)
        canonical[code] = min_code

    unique_vals = np.unique(canonical)
    remap = {old: new for new, old in enumerate(unique_vals)}
    lut = np.array([remap[v] for v in canonical], dtype=np.int32)

    return lut, len(unique_vals)

def _encode_codes(
    codes: np.ndarray,
    p: int,
    method: LTPMethod,
) -> tuple[np.ndarray, int]:
    """Encode raw binary codes according to the requested method."""
    if method == "default":
        return codes.astype(np.int32, copy=False), 1 << p

    if method == "ror":
        lut, n_bins = _ror_lookup(p)
        encoded = lut[codes.astype(np.int32, copy=False)]
        return encoded, n_bins

    if method == "uniform":
        lut, n_bins = _uniform_lookup(p)
        return lut[codes.astype(np.int32, copy=False)], n_bins

    raise ValueError(f"Unsupported method {method!r}. Use 'default', 'ror', or 'uniform'.")


def _normalized_histogram(
    codes: np.ndarray,
    n_bins: int,
    mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute an L1-normalized histogram for encoded code values."""
    flat = codes.ravel()
    if mask is not None:
        if mask.shape != codes.shape:
            raise ValueError(
                f"`mask` shape must match codes shape, got {mask.shape} vs {codes.shape}."
            )
        flat = flat[np.asarray(mask, dtype=bool).ravel()]

    hist = np.bincount(flat, minlength=n_bins).astype(np.float32)
    hist /= hist.sum() + eps
    return hist


def local_ternary_pattern(
    gray: np.ndarray,
    p: int = 8,
    r: float = 1.0,
    threshold: float = 5.0,
    *,
    method: LTPMethod = "uniform",
    mask: np.ndarray | None = None,
    normalize: bool = True,
) -> LTPResult:
    """Compute Local Ternary Pattern features for a grayscale image.

    This implementation uses the standard split-LTP formulation:
    - Positive pattern: neighbor >= center + threshold
    - Negative pattern: neighbor <= center - threshold

    The final feature vector is:
        concat(histogram(positive_codes), histogram(negative_codes))

    Args:
        gray: 2D grayscale image.
        p: Number of circular neighbors.
        r: Sampling radius in pixels.
        threshold: Ternary threshold. For uint8 images, values like 3-10 are common.
        method: Encoding method:
            - "default": raw binary codes
            - "ror": rotation-invariant via minimum circular bit rotation
            - "uniform": uniform-pattern encoding
        mask: Optional boolean mask restricting which pixels contribute to histograms.
        normalize: Whether to L1-normalize the concatenated feature vector.

    Returns:
        LTPResult containing positive codes, negative codes, and concatenated histogram.
    """
    gray_f = _validate_gray_image(gray)

    codes_pos_raw, codes_neg_raw = _compute_ltp_codes(
        gray=gray_f,
        p=p,
        r=r,
        threshold=threshold,
    )

    codes_pos, n_bins_pos = _encode_codes(codes_pos_raw, p=p, method=method)
    codes_neg, n_bins_neg = _encode_codes(codes_neg_raw, p=p, method=method)

    hist_pos = _normalized_histogram(codes_pos, n_bins=n_bins_pos, mask=mask)
    hist_neg = _normalized_histogram(codes_neg, n_bins=n_bins_neg, mask=mask)

    histogram = np.concatenate([hist_pos, hist_neg]).astype(np.float32, copy=False)

    if normalize:
        histogram /= histogram.sum() + 1e-6

    return LTPResult(
        codes_pos=codes_pos,
        codes_neg=codes_neg,
        histogram=histogram,
    )