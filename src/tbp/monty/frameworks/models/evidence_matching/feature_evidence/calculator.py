# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import weakref
from typing import Protocol

import numpy as np

from tbp.monty.frameworks.utils.sensor_processing import LTP_PIXEL_STATS_KEY

# Per-node histogram statistics (the elementwise square root of each stored histogram
# and its L1 norm) needed by the Hellinger distance. These depend only on the stored
# graph and are reused on every matching step, so they are cached rather than
# recomputed. Graph memory rebuilds feature arrays wholesale instead of mutating them
# in place, which makes the array's identity a safe cache key; a finalizer drops the
# entry once the array is garbage collected, so a recycled `id()` can never produce a
# stale hit.
_HISTOGRAM_STATS_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
_HISTOGRAM_STATS_FINALIZERS: dict[int, weakref.finalize] = {}


def _evict_histogram_stats(array_id: int) -> None:
    for key in [key for key in _HISTOGRAM_STATS_CACHE if key[0] == array_id]:
        del _HISTOGRAM_STATS_CACHE[key]
    _HISTOGRAM_STATS_FINALIZERS.pop(array_id, None)


def _stored_histogram_stats(
    channel_feature_array: np.ndarray, start_idx: int, end_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Square roots and L1 norms of the stored histograms in the given column range.

    Args:
        channel_feature_array: Stored features for every node in the graph.
        start_idx: First column of the histogram feature.
        end_idx: One past the last column of the histogram feature.

    Returns:
        The elementwise square root of each node's stored histogram, shape
        `(n_nodes, n_bins)`, and each stored histogram's L1 norm, shape `(n_nodes,)`.
    """
    array_id = id(channel_feature_array)
    cache_key = (array_id, start_idx, end_idx)
    cached = _HISTOGRAM_STATS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Histograms are stored as float32 by the sensor module; matching that precision
    # here keeps the distances identical to the values the previous per-node
    # cv2.compareHist implementation produced.
    stored = channel_feature_array[:, start_idx:end_idx].astype(np.float32)
    stats = (np.sqrt(stored, dtype=np.float64), stored.sum(axis=1, dtype=np.float64))

    _HISTOGRAM_STATS_CACHE[cache_key] = stats
    if array_id not in _HISTOGRAM_STATS_FINALIZERS:
        _HISTOGRAM_STATS_FINALIZERS[array_id] = weakref.finalize(
            channel_feature_array, _evict_histogram_stats, array_id
        )
    return stats


class FeatureEvidenceCalculator(Protocol):
    @staticmethod
    def calculate(
        channel_feature_array: np.ndarray,
        channel_feature_order: list[str],
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
    ) -> np.ndarray: ...


class DefaultFeatureEvidenceCalculator:
    SKIP_FEATURES = frozenset(
        {"pose_vectors", "pose_fully_defined", LTP_PIXEL_STATS_KEY}
    )
    CIRCULAR_FEATURES = frozenset({"hsv"})
    CATEGORICAL_FEATURES = frozenset({"object_id"})
    # `ltp_rgb` stores one histogram per color channel back to back. It is scored as a
    # single histogram rather than channel by channel because the Bhattacharyya
    # coefficient is additive across bins and the channels contribute equally many
    # pixels: the Hellinger distance over the concatenation is exactly the
    # root-mean-square of the three per-channel Hellinger distances, so the channels
    # are already compared independently and then pooled.
    HISTOGRAM_FEATURES = frozenset({"ltp", "ltp_rgb"})
    # Histogram features subject to the patch-intensity reliability gate below. Only
    # the grayscale `ltp` qualifies: the gate is defined in terms of the grayscale
    # patch statistics, which `ltp_rgb` does not report.
    GATED_HISTOGRAM_FEATURES = frozenset({"ltp"})
    CIRCULAR_RANGE = 1

    # When the patch that produced an LTP histogram is dark (low mean pixel
    # intensity), abnormally bright (high mean pixel intensity, e.g. blown-out or
    # specular highlights), or nearly uniform (low pixel-intensity variance), the
    # texture signal is dominated by sensor noise or saturation. In that regime the
    # LTP evidence is unreliable, so its feature weight is forced to 0 (rather than
    # the configured value) for that observation. Intensities are in the 0-255
    # grayscale range.
    LTP_DARK_MEAN_INTENSITY_THRESHOLD = 60.0
    LTP_BRIGHT_MEAN_INTENSITY_THRESHOLD = 230.0
    LTP_LOW_INTENSITY_VARIANCE_THRESHOLD = 400.0

    @classmethod
    def calculate(
        cls,
        channel_feature_array: np.ndarray,
        channel_feature_order: list[str],
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
    ) -> np.ndarray:
        """Calculate the feature evidence for all nodes stored in a graph.

        For each node, compares the stored features against the observed
        query features and returns a score in `[0, 1]`: 1 for a perfect
        match, decaying to 0 once the difference exceeds the per-feature
        tolerance. Nodes with missing stored values for a feature receive
        NaN evidence.

        Args:
            channel_feature_array: Stored features for every node in the
                graph, shape `(n_nodes, n_columns)`. Columns follow the
                layout given by `channel_feature_order`.
            channel_feature_order: Feature names in the order they appear
                across the columns of `channel_feature_array`.
            channel_feature_weights: Per-feature weights used to combine
                per-column evidence into a single per-node score.
            channel_query_features: Observed feature values to compare
                against the stored features, keyed by feature name.
            channel_tolerances: Per-feature tolerance, the largest
                difference that still produces non-zero evidence.

        Returns:
            The feature evidence for all nodes, shape `(n_nodes,)`.
        """
        # A histogram feature spans one stored column per bin, but it is conceptually a
        # single feature: every bin shares the same per-node Hellinger distance. Rather
        # than replicating that distance across all of its bin columns, each histogram
        # is scored into a single column here. The two layouts are equivalent under the
        # weighted average (n_bins columns each carrying weight w/n_bins contribute
        # exactly as much as one column carrying weight w), but the compact layout
        # keeps the elementwise math below proportional to the number of features
        # rather than to the number of histogram bins, which dominates otherwise.
        scored_layout = cls._scored_layout(
            channel_feature_order, channel_query_features
        )
        n_scored_cols = sum(width for *_, width in scored_layout)

        n_nodes = channel_feature_array.shape[0]
        tolerance_list = np.full(n_scored_cols, np.nan)
        feature_weight_list = np.full(n_scored_cols, np.nan)
        feature_differences = np.empty((n_nodes, n_scored_cols))
        scored_cols = 0

        for feature, stored_start, stored_end, width in scored_layout:
            scored_start = scored_cols
            scored_end = scored_cols + width
            scored_cols = scored_end

            tolerance_list[scored_start:scored_end] = channel_tolerances[feature]
            query_feature = np.atleast_1d(
                np.asarray(channel_query_features[feature], dtype=np.float64)
            )
            stored = channel_feature_array[:, stored_start:stored_end]

            if feature in cls.CIRCULAR_FEATURES:
                # H is circular, S and V are numeric
                feature_weight_list[scored_start:scored_end] = channel_feature_weights[
                    feature
                ]
                hue_stored = stored[:, :1]
                hue_query = query_feature[:1]
                feature_differences[:, scored_start : scored_start + 1] = np.min(
                    [
                        np.abs(cls.CIRCULAR_RANGE + hue_stored - hue_query),
                        np.abs(hue_stored - hue_query),
                        np.abs(hue_stored - (hue_query + cls.CIRCULAR_RANGE)),
                    ],
                    axis=0,
                )
                feature_differences[:, scored_start + 1 : scored_end] = np.abs(
                    stored[:, 1:] - query_feature[1:]
                )
            elif feature in cls.CATEGORICAL_FEATURES:
                feature_weight_list[scored_start:scored_end] = channel_feature_weights[
                    feature
                ]
                feature_differences[:, scored_start:scored_end] = (
                    stored != query_feature
                )
            elif feature in cls.HISTOGRAM_FEATURES:
                # The grayscale LTP texture signal is unreliable when the observed
                # patch is too dark and uniform, so drop its weight to 0 for this
                # observation rather than using the configured value.
                unreliable = feature in cls.GATED_HISTOGRAM_FEATURES and (
                    cls._is_unreliable_ltp_observation(channel_query_features)
                )
                feature_weight_list[scored_start:scored_end] = (
                    0.0 if unreliable else channel_feature_weights[feature]
                )
                feature_differences[:, scored_start] = cls._hellinger_distances(
                    channel_feature_array, stored_start, stored_end, query_feature
                )
            else:
                feature_weight_list[scored_start:scored_end] = channel_feature_weights[
                    feature
                ]
                feature_differences[:, scored_start:scored_end] = np.abs(
                    stored - query_feature
                )

        # any difference < tolerance should be positive evidence
        # any difference >= tolerance should be 0 evidence
        feature_evidence = np.clip(tolerance_list - feature_differences, 0, np.inf)
        # normalize evidence to be in [0, 1]
        feature_evidence = feature_evidence / tolerance_list
        # If every feature weight is 0 (e.g. LTP was the only matched feature and was
        # zeroed out for an unreliable observation), there is no feature evidence to
        # contribute, so return zeros instead of dividing by a zero total weight.
        if not np.any(feature_weight_list):
            return np.zeros(n_nodes)
        return np.average(feature_evidence, weights=feature_weight_list, axis=1)

    @classmethod
    def _scored_layout(
        cls, channel_feature_order: list[str], channel_query_features: dict
    ) -> list[tuple[str, int, int, int]]:
        """Map each matched feature to its stored columns and its scored width.

        Args:
            channel_feature_order: Feature names in the order they appear across the
                columns of the stored feature array.
            channel_query_features: Observed feature values, used to size each feature.

        Returns:
            One `(feature, stored_start, stored_end, scored_width)` entry per matched
            feature. `scored_width` is 1 for histogram features (which collapse to a
            single distance column) and equal to the feature's length otherwise.
        """
        layout = []
        stored_start = 0
        for feature in channel_feature_order:
            if feature in cls.SKIP_FEATURES:
                continue
            query_feature = channel_query_features[feature]
            feature_length = (
                len(query_feature) if hasattr(query_feature, "__len__") else 1
            )
            stored_end = stored_start + feature_length
            scored_width = 1 if feature in cls.HISTOGRAM_FEATURES else feature_length
            layout.append((feature, stored_start, stored_end, scored_width))
            stored_start = stored_end
        return layout

    @classmethod
    def _hellinger_distances(
        cls,
        channel_feature_array: np.ndarray,
        stored_start: int,
        stored_end: int,
        query_hist: np.ndarray,
    ) -> np.ndarray:
        """Hellinger distance between the query histogram and every stored histogram.

        Hellinger is bounded in `[0, 1]` and symmetric, which makes the tolerance easy
        to interpret and avoids the unbounded blow-up that an asymmetric chi-square
        produces on the sparse, spiky LTP histograms (small stored bins dominate the
        score). For histograms `P` (stored) and `Q` (query) the distance is

            sqrt(1 - sum_i sqrt(P_i * Q_i) / sqrt(sum(P) * sum(Q)))

        The numerator factors into a matrix-vector product between the stored square
        roots and the query's square roots, so every node is scored in one pass.

        Args:
            channel_feature_array: Stored features for every node in the graph.
            stored_start: First stored column of the histogram.
            stored_end: One past the last stored column of the histogram.
            query_hist: The observed histogram.

        Returns:
            The per-node distance to the query histogram, shape `(n_nodes,)`.
        """
        sqrt_stored, stored_l1 = _stored_histogram_stats(
            channel_feature_array, stored_start, stored_end
        )
        # Histograms are stored as float32 by the sensor module, so the query is cast
        # to match before the distance is computed at float64 precision.
        query_hist = query_hist.astype(np.float32)
        bhattacharyya_coefficient = sqrt_stored @ np.sqrt(query_hist, dtype=np.float64)

        # The normalizing term degenerates when a histogram sums to zero (e.g. an
        # all-zero stored histogram). There the scale falls back to 1, putting the node
        # at the maximum distance of 1 rather than dividing by zero.
        l1_product = np.abs(stored_l1 * query_hist.sum(dtype=np.float64))
        scale = np.ones_like(l1_product)
        np.divide(
            1.0,
            np.sqrt(l1_product),
            out=scale,
            where=l1_product > np.finfo(np.float32).eps,
        )
        return np.sqrt(np.maximum(1.0 - bhattacharyya_coefficient * scale, 0.0))

    @classmethod
    def _is_unreliable_ltp_observation(cls, channel_query_features: dict) -> bool:
        """Whether the LTP texture signal should be discounted for this observation.

        The patch that produced the LTP histogram is considered to carry too
        little meaningful texture signal when its mean pixel intensity is too low
        (dark) or too high (abnormally bright, e.g. saturated/specular), or when
        its pixel-intensity variance falls below its threshold (too uniform).

        Args:
            channel_query_features: Observed feature values for the channel,
                optionally including the LTP patch intensity statistics under
                `LTP_PIXEL_STATS_KEY` as `[mean, variance]`.

        Returns:
            True if the LTP evidence should be assigned zero weight.
        """
        stats = channel_query_features.get(LTP_PIXEL_STATS_KEY)
        if stats is None:
            return False
        mean_intensity = float(stats[0])
        intensity_variance = float(stats[1])

        return (
            mean_intensity < cls.LTP_DARK_MEAN_INTENSITY_THRESHOLD
            or mean_intensity > cls.LTP_BRIGHT_MEAN_INTENSITY_THRESHOLD
            or intensity_variance < cls.LTP_LOW_INTENSITY_VARIANCE_THRESHOLD
        )
