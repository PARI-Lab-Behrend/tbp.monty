# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Shared helpers for analyzing and plotting LTP texture-histogram distances.

A Monty object model stores a normalized ``ltp`` (Local Ternary Pattern)
histogram per graph node. This module measures how far apart those histograms
are under a chosen distance metric, split into two regimes:

    within-object : pairs of nodes belonging to the *same* object
    across-object : pairs of nodes belonging to *different* objects

A discriminative metric keeps within-object distances small and across-object
distances large (little overlap), so this is the calibration tool for the ``ltp``
tolerance/weight (see ``LTP_BRANCH_COMPARISON_ANALYSIS.md`` §4b).

The tolerance is calibrated against the evidence Monty actually accumulates, not
against a hard distance threshold. Monty never asks "is this distance below
``tol``?" -- :class:`DefaultFeatureEvidenceCalculator` maps the distance through
the ramp ``max(0, 1 - d/tol)`` and *adds* the result to the pose evidence. Since
feature evidence is non-negative, a mismatching node is never penalized, and all
of LTP's discriminative work is done by the *gap* between the evidence a correct
node accrues and what a wrong one accrues. :func:`best_tolerance` maximizes that
gap; :func:`best_threshold` (Youden's J) is retained only as a reference, and
picks a systematically too-low tolerance that clips correct-object nodes to zero.

The ``chisq`` and ``hellinger`` metrics are the **actual Monty matching
functions** (:class:`DefaultFeatureEvidenceCalculator` /
:class:`HellingerFeatureEvidenceCalculator`), so the distances plotted here are
exactly what Monty computes during matching. The remaining metrics (``l1``,
``l2``, ``intersection``, ``bhattacharyya``, ``correlation``, ``kldiv``,
``jensen_shannon``) are extras for comparison; add your own to :data:`METRICS`
-- any
``f(stored_hists[n, bins], query_hist[bins]) -> distances[n]`` works (the same
signature as ``FeatureEvidenceCalculator.histogram_distance``).

This module is consumed both by ``scripts/plot_ltp_distances.py`` (offline, loads
a ``model.pt``) and by
:class:`~tbp.monty.frameworks.experiments.analysis_experiments.MontyLTPDistanceExperiment`
(runs via ``run.py`` against the live graph memory).
"""

from __future__ import annotations

import re

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Distance metrics. Each maps stored histograms (n_nodes, n_bins) and a single
# query histogram (n_bins,) to per-node distances (n_nodes,). All assume
# normalized histograms (each row sums to ~1).
# --------------------------------------------------------------------------- #
def _l1(stored: np.ndarray, query: np.ndarray) -> np.ndarray:
    """L1 / Manhattan (= 2x total variation) distance."""
    return np.abs(stored - query).sum(axis=1)


def _l2(stored: np.ndarray, query: np.ndarray) -> np.ndarray:
    """L2 / Euclidean distance."""
    return np.linalg.norm(stored - query, axis=1)


def _intersection(stored: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Histogram-intersection distance: ``1 - sum(min(h1, h2))`` (in [0, 1])."""
    return 1.0 - np.minimum(stored, query).sum(axis=1)


def _bhattacharyya(stored: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Bhattacharyya distance ``sqrt(1 - sum(sqrt(h1 h2)))`` (in [0, 1])."""
    bc = np.sqrt(stored * query).sum(axis=1)
    return np.sqrt(np.clip(1.0 - bc, 0.0, None))


def _correlation(stored: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Correlation distance ``1 - corr`` (in [0, 2]); matches cv2.HISTCMP_CORREL sign."""
    s = stored - stored.mean(axis=1, keepdims=True)
    q = query - query.mean()
    num = (s * q).sum(axis=1)
    den = np.sqrt((s * s).sum(axis=1) * (q * q).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(den > 0, num / den, 0.0)
    return 1.0 - corr


def _kldiv(stored: np.ndarray, query: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """KL divergence ``D(query || stored) = sum query log(query / stored)``.

    Asymmetric and unbounded ``[0, inf)``; not a true metric. Bins are smoothed by
    ``eps`` so that a stored bin with zero mass where the query has mass yields a
    large (rather than infinite) penalty.
    """
    q = query + eps
    s = stored + eps
    return np.where(query > 0, query * np.log(q / s), 0.0).sum(axis=1)


def _jensen_shannon(stored: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Jensen-Shannon distance (sqrt of the JS divergence, log base 2), in [0, 1].

    Symmetric, bounded, and a true metric. ``JSD = 0.5 D(P||M) + 0.5 D(Q||M)`` with
    ``M = (P + Q) / 2``; the distance is ``sqrt(JSD)``. Zero-mass bins contribute 0
    (``M`` is non-zero wherever either input is), so no smoothing is needed.
    """
    m = 0.5 * (stored + query)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_stored = np.where(stored > 0, stored * np.log2(stored / m), 0.0).sum(axis=1)
        kl_query = np.where(query > 0, query * np.log2(query / m), 0.0).sum(axis=1)
    jsd = 0.5 * kl_stored + 0.5 * kl_query
    return np.sqrt(np.clip(jsd, 0.0, None))


def _hellinger(stored: np.ndarray, query: np.ndarray) -> np.ndarray:
    """cv2 Bhattacharyya/Hellinger distance -- the actual Monty LTP matching function.

    :class:`DefaultFeatureEvidenceCalculator` scores LTP histograms with
    ``cv2.compareHist(..., cv2.HISTCMP_BHATTACHARYYA)``, so the distances plotted
    here are exactly what Monty computes during recognition on this branch.
    """
    q = query.astype(np.float32)
    return np.array(
        [
            cv2.compareHist(s.astype(np.float32), q, cv2.HISTCMP_BHATTACHARYYA)
            for s in stored
        ]
    )


def _chisq(stored: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Asymmetric chi-square distance ``sum((stored - query)^2 / stored)``.

    Kept for comparison with the Hellinger score Monty actually uses; bins with
    zero stored mass contribute 0 rather than diverging.
    """
    diff = stored - query
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(stored > 0, diff * diff / stored, 0.0)
    return terms.sum(axis=1)


# ``hellinger`` is the real Monty matching function so the plotted distances are
# exactly those used during recognition.
METRICS = {
    "chisq": _chisq,
    "hellinger": _hellinger,
    "l1": _l1,
    "l2": _l2,
    "intersection": _intersection,
    "bhattacharyya": _bhattacharyya,
    "correlation": _correlation,
    "kldiv": _kldiv,
    "jensen_shannon": _jensen_shannon,
}


# --------------------------------------------------------------------------- #
# Histogram extraction.
# --------------------------------------------------------------------------- #
def extract_histograms(
    models: dict[str, dict[str, object]], feature: str = "ltp"
) -> dict[str, np.ndarray]:
    """Pull normalized per-object histograms out of stored object models.

    Args:
        models: Mapping ``{object_id: {input_channel: object_model}}`` -- exactly
            the structure of ``EvidenceGraphMemory.models_in_memory`` (and of
            ``state["lm_dict"][lm]["graph_memory"]`` in a saved ``model.pt``).
        feature: Histogram feature name (default ``"ltp"``).

    Returns:
        ``{object_id: array(n_nodes, n_bins)}``.

    Raises:
        KeyError: If ``feature`` is not stored on a model.
        ValueError: If no histograms are found.
    """
    try:
        import torch
    except ImportError:  # torch is always present in practice; guard for typing.
        torch = None  # type: ignore[assignment]

    hists: dict[str, np.ndarray] = {}
    for object_id, channels in models.items():
        per_channel = []
        for model in channels.values():
            mapping = dict(model.feature_mapping)
            if feature not in mapping:
                raise KeyError(
                    f"Feature {feature!r} not in {object_id} "
                    f"(have: {sorted(mapping)})"
                )
            start, end = mapping[feature]
            x = model.x
            if torch is not None and isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            else:
                x = np.asarray(x)
            h = x[:, start:end].astype(np.float64)
            # Re-normalize defensively; stored histograms should already sum to ~1.
            row_sums = h.sum(axis=1, keepdims=True)
            h = np.divide(h, row_sums, out=np.zeros_like(h), where=row_sums > 0)
            per_channel.append(h)
        if per_channel:
            hists[object_id] = np.vstack(per_channel)
    if not hists:
        raise ValueError("No object models with histograms found.")
    return hists


def histograms_from_model_file(
    model_dir, feature: str = "ltp", lm_id: int = 0
) -> dict[str, np.ndarray]:
    """Load per-object histograms from a saved ``model.pt`` (offline path)."""
    from pathlib import Path

    import torch

    model_dir = Path(model_dir).expanduser()
    model_path = model_dir if model_dir.name == "model.pt" else model_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"No model.pt found at {model_path}")
    state = torch.load(model_path, weights_only=False)
    return extract_histograms(state["lm_dict"][lm_id]["graph_memory"], feature)


def build_groups(
    object_ids, group_regex: str | None
) -> dict[str, str] | None:
    """Map each object id to a shape-group key via ``group_regex`` (or ``None``).

    The matched substring is the group key; objects with no match form their own
    single-member group. Used to add within-group/across-group regimes (e.g. to
    isolate same-shape reskins).
    """
    if not group_regex:
        return None
    pattern = re.compile(group_regex)
    group_of = {}
    for obj in object_ids:
        m = pattern.search(obj)
        group_of[obj] = m.group(0) if m else obj
    return group_of


# --------------------------------------------------------------------------- #
# Distance sampling.
# --------------------------------------------------------------------------- #
def subsample(
    hists: dict[str, np.ndarray], max_nodes: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Cap each object to ``max_nodes`` randomly chosen nodes (keeps pairs tractable)."""
    out = {}
    for obj, h in hists.items():
        if max_nodes and h.shape[0] > max_nodes:
            idx = rng.choice(h.shape[0], size=max_nodes, replace=False)
            h = h[idx]
        out[obj] = h
    return out


def collect_distances(
    hists: dict[str, np.ndarray],
    metric,
    rng: np.random.Generator,
    n_queries: int,
    group_of: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Sample distances by drawing query nodes and comparing to all other nodes.

    Returns distance arrays keyed by regime: ``within_object``, ``across_object``,
    and (when ``group_of`` is given) ``within_group`` (same shape group, different
    object -- the hard reskin case) and ``across_group``.
    """
    objects = list(hists.keys())
    all_h = np.vstack([hists[o] for o in objects])
    obj_labels = np.concatenate(
        [np.full(hists[o].shape[0], i) for i, o in enumerate(objects)]
    )
    node_group = np.array([])
    if group_of is not None:
        group_ids = {g: i for i, g in enumerate(sorted(set(group_of.values())))}
        node_group = np.concatenate(
            [np.full(hists[o].shape[0], group_ids[group_of[o]]) for o in objects]
        )

    n_total = all_h.shape[0]
    n_queries = min(n_queries, n_total)
    query_idx = rng.choice(n_total, size=n_queries, replace=False)

    buckets: dict[str, list[np.ndarray]] = {
        "within_object": [],
        "across_object": [],
    }
    if group_of is not None:
        buckets["within_group"] = []
        buckets["across_group"] = []

    for qi in query_idx:
        dists = metric(all_h, all_h[qi])
        same_obj = obj_labels == obj_labels[qi]
        same_obj[qi] = False  # exclude self-distance
        buckets["within_object"].append(dists[same_obj])
        buckets["across_object"].append(dists[obj_labels != obj_labels[qi]])
        if group_of is not None:
            same_grp = node_group == node_group[qi]
            buckets["within_group"].append(dists[same_grp & ~same_obj])
            buckets["across_group"].append(dists[~same_grp])

    return {k: np.concatenate(v) if v else np.array([]) for k, v in buckets.items()}


def best_threshold(
    within: np.ndarray, across: np.ndarray
) -> dict[str, float] | None:
    """Distance threshold that best separates ``within`` from ``across``.

    A node is classified "same object" when its distance is ``<= threshold``, and
    the returned threshold maximizes Youden's J (``TPR + TNR - 1``, = balanced
    accuracy), which is robust to the heavy class imbalance (far more across- than
    within-object pairs).

    .. warning::
       This is **not** the right way to pick the ``ltp`` tolerance, and is kept
       only as a reference statistic. It scores a hard classifier, but Monty never
       thresholds the distance: :class:`DefaultFeatureEvidenceCalculator` maps it
       through the ramp ``max(0, 1 - d/tol)`` and *adds* the result to the pose
       evidence. Feature evidence is non-negative, so a mismatching node is never
       penalized -- all of LTP's discriminative work is done by the *gap* between
       the evidence a correct node accrues and what a wrong one accrues. Youden's J
       systematically picks a tolerance far below the gap-maximizing one, which
       clips a large fraction of correct-object nodes to exactly zero evidence and
       throws that signal away. Use :func:`best_tolerance` instead.

    Returns ``None`` if either group is empty.
    """
    if within.size == 0 or across.size == 0:
        return None
    sw = np.sort(within)
    sa = np.sort(across)
    candidates = np.unique(np.concatenate([within, across]))
    tpr = np.searchsorted(sw, candidates, side="right") / sw.size
    tnr = (sa.size - np.searchsorted(sa, candidates, side="right")) / sa.size
    j = tpr + tnr - 1.0
    k = int(np.argmax(j))
    return {
        "threshold": float(candidates[k]),
        "balanced_acc": float((tpr[k] + tnr[k]) / 2.0),
        "tpr": float(tpr[k]),
        "tnr": float(tnr[k]),
        "youden_j": float(j[k]),
    }


# --------------------------------------------------------------------------- #
# Tolerance calibration against the evidence Monty actually accumulates.
# --------------------------------------------------------------------------- #
def feature_evidence(distances: np.ndarray, tolerance: float) -> np.ndarray:
    """Feature evidence Monty derives from histogram distances at a given tolerance.

    This is exactly the ramp :class:`DefaultFeatureEvidenceCalculator` applies:
    ``clip(tolerance - d, 0, inf) / tolerance``, i.e. 1 at zero distance, decaying
    linearly to 0 at ``d >= tolerance``.
    """
    return np.clip(tolerance - distances, 0.0, np.inf) / tolerance


def default_tolerance_grid(
    within: np.ndarray, across: np.ndarray, n: int = 200
) -> np.ndarray:
    """Candidate tolerances spanning the observed distance range.

    The grid is metric-agnostic (Hellinger is bounded in ``[0, 1]`` but chi-square
    and KL are not), so it runs up to the 99.5th percentile of the pooled
    distances.
    """
    pooled = np.concatenate([within, across])
    hi = float(np.percentile(pooled, 99.5))
    hi = hi if hi > 0 else 1.0
    return np.linspace(hi / n, hi, n)


def tolerance_sweep(
    within: np.ndarray, across: np.ndarray, tolerances: np.ndarray | None = None
) -> list[dict[str, float]]:
    """Score each candidate tolerance by the evidence it produces.

    For every tolerance this reports the mean feature evidence a same-object node
    receives, the mean a different-object node receives, and the two quantities
    that determine how well Monty can tell them apart:

    ``gap``
        ``E[ev | same object] - E[ev | different object]``. Feature evidence is
        added to pose evidence and is never negative, so this gap is the entire
        per-step contribution LTP makes to separating the correct hypothesis from
        its competitors.
    ``d_prime``
        The gap normalized by the pooled standard deviation. Evidence is summed
        over many matching steps, so the separation after ``n`` steps grows like
        ``sqrt(n) * d_prime``; this is the better objective when the two evidence
        distributions have very different spreads.

    ``frac_within_clipped`` is the fraction of same-object nodes whose distance
    exceeds the tolerance and are therefore clipped to *exactly* zero evidence --
    indistinguishable from a completely mismatching node. Clipping is the only
    irreversible step in the pipeline, and it is what makes an aggressively low
    tolerance harmful.
    """
    if tolerances is None:
        tolerances = default_tolerance_grid(within, across)

    rows = []
    for tol in tolerances:
        ev_within = feature_evidence(within, tol)
        ev_across = feature_evidence(across, tol)
        gap = float(ev_within.mean() - ev_across.mean())
        pooled_sd = float(np.sqrt(0.5 * (ev_within.var() + ev_across.var())))
        rows.append(
            {
                "tolerance": float(tol),
                "ev_within": float(ev_within.mean()),
                "ev_across": float(ev_across.mean()),
                "gap": gap,
                "d_prime": gap / pooled_sd if pooled_sd > 0 else 0.0,
                "frac_within_clipped": float((within >= tol).mean()),
            }
        )
    return rows


def best_tolerance(
    within: np.ndarray, across: np.ndarray, tolerances: np.ndarray | None = None
) -> dict[str, object] | None:
    """Tolerance that maximizes the evidence gap between same- and other-object nodes.

    This is the value to use as the ``ltp`` tolerance. The returned dict carries the
    gap-optimal row, the ``d_prime``-optimal row (usually close by), and the full
    sweep. Returns ``None`` if either group is empty.
    """
    if within.size == 0 or across.size == 0:
        return None
    sweep = tolerance_sweep(within, across, tolerances)
    return {
        "by_gap": max(sweep, key=lambda r: r["gap"]),
        "by_d_prime": max(sweep, key=lambda r: r["d_prime"]),
        "sweep": sweep,
    }


def _print_tolerance_sweep(name: str, within: np.ndarray, across: np.ndarray) -> float:
    """Print the evidence-vs-tolerance table; return the gap-optimal tolerance."""
    best = best_tolerance(within, across)
    sweep = best["sweep"]

    # Show a readable subset of the grid rather than all 200 rows.
    step = max(1, len(sweep) // 12)
    shown = sweep[::step]
    for row in (best["by_gap"], best["by_d_prime"]):
        if row not in shown:
            shown.append(row)
    shown.sort(key=lambda r: r["tolerance"])

    print(f"\n[{name}] feature evidence vs. ltp tolerance (within_object vs across)")
    header = (
        f"{'tolerance':>10}{'ev_within':>12}{'ev_across':>12}"
        f"{'gap':>10}{'d_prime':>10}{'%within@0':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in shown:
        marks = []
        if row is best["by_gap"]:
            marks.append("<- max gap")
        if row is best["by_d_prime"]:
            marks.append("<- max d'")
        print(
            f"{row['tolerance']:>10.4f}{row['ev_within']:>12.4f}"
            f"{row['ev_across']:>12.4f}{row['gap']:>10.4f}"
            f"{row['d_prime']:>10.4f}{row['frac_within_clipped'] * 100:>11.1f}%"
            f"  {' '.join(marks)}"
        )

    print(
        f"[{name}] recommended ltp tolerance = {best['by_gap']['tolerance']:.4f} "
        f"(max gap {best['by_gap']['gap']:.4f}); "
        f"max d' at {best['by_d_prime']['tolerance']:.4f}"
    )
    return best["by_gap"]["tolerance"]


def summarize(name: str, dist_by_regime: dict[str, np.ndarray]) -> float | None:
    """Print per-regime distance stats and calibrate the tolerance against evidence.

    Returns the evidence-gap-optimal within-object-vs-across-object tolerance (or
    ``None``) so the caller can mark it on the plot. Youden's J threshold is also
    printed for reference, but it is *not* the tolerance to use -- see
    :func:`best_threshold`.
    """
    pct = [25, 50, 75, 90, 95]
    header = f"{'regime':<16}{'count':>10}{'mean':>10}" + "".join(
        f"{'p' + str(p):>10}" for p in pct
    )
    print(f"\n[{name}] distance distribution by regime")
    print(header)
    print("-" * len(header))
    for regime, d in dist_by_regime.items():
        if d.size == 0:
            continue
        cells = [f"{d.mean():>10.3f}"] + [
            f"{np.percentile(d, p):>10.3f}" for p in pct
        ]
        print(f"{regime:<16}{d.size:>10}{''.join(cells)}")

    within = dist_by_regime.get("within_object", np.array([]))
    print(
        f"\n[{name}] Youden-J threshold (reference only; scores a hard classifier, "
        "not Monty's evidence ramp)"
    )
    for regime, neg in dist_by_regime.items():
        if regime == "within_object":
            continue
        res = best_threshold(within, neg)
        if res is None:
            continue
        print(
            f"  vs {regime:<14} threshold={res['threshold']:.4f}  "
            f"balanced_acc={res['balanced_acc']:.3f}  "
            f"(within<= t: {res['tpr']:.3f}, {regime}> t: {res['tnr']:.3f})"
        )

    across = dist_by_regime.get("across_object", np.array([]))
    if within.size == 0 or across.size == 0:
        return None
    return _print_tolerance_sweep(name, within, across)


# --------------------------------------------------------------------------- #
# Plotting.
# --------------------------------------------------------------------------- #
REGIME_COLORS = {
    "within_object": "tab:green",
    "within_group": "tab:orange",
    "across_group": "tab:red",
    "across_object": "tab:blue",
}


def _plot_metric(ax, name, dist_by_regime, bins, threshold=None, xmax=None) -> None:
    all_vals = np.concatenate([d for d in dist_by_regime.values() if d.size])
    if xmax is not None:
        # Explicit x-axis upper bound: bins span [0, xmax] and out-of-range
        # distances are dropped (not piled into the edge bin).
        hi = xmax if xmax > 0 else 1.0
        clip = False
    else:
        # Default: clip the x-range to the 99.5th pct so unbounded metrics
        # (chi-square) stay readable instead of being squashed by a few outliers.
        hi = np.percentile(all_vals, 99.5)
        hi = hi if hi > 0 else 1.0
        clip = True
    edges = np.linspace(0, hi, bins + 1)
    for regime, d in dist_by_regime.items():
        if d.size == 0:
            continue
        ax.hist(
            np.clip(d, edges[0], edges[-1]) if clip else d,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"{regime} (n={d.size})",
            color=REGIME_COLORS.get(regime),
        )
    ax.set_xlim(0, hi)
    if threshold is not None:
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label=f"evidence-optimal tol = {threshold:.4f}",
        )
    ax.set_title(name)
    ax.set_xlabel("distance")
    ax.set_ylabel("density")
    ax.legend(fontsize="small")


def plot_distance_distributions(
    results: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float | None],
    out_path,
    title: str,
    bins: int = 60,
    xmax: float | None = None,
) -> None:
    """Render one subplot per metric (within/across histograms + best threshold).

    ``xmax`` sets an explicit upper bound for every subplot's x-axis; when ``None``
    each subplot auto-scales to its 99.5th percentile.
    """
    import math

    import matplotlib

    matplotlib.use("Agg")  # write to file; no display needed
    import matplotlib.pyplot as plt

    metric_names = list(results)
    n = len(metric_names)
    # Near-square grid so the figure stays balanced for any metric count.
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    # Lay out on a doubled column grid so a partial last row can be centered
    # by half-cell offsets (each subplot spans two sub-columns).
    fig = plt.figure(figsize=(5.5 * ncols, 4.5 * nrows))
    gs = fig.add_gridspec(nrows, 2 * ncols)
    for i, name in enumerate(metric_names):
        row, col = divmod(i, ncols)
        items_in_row = min(ncols, n - row * ncols)
        offset = ncols - items_in_row  # half-cells needed to center this row
        start = offset + 2 * col
        ax = fig.add_subplot(gs[row, start : start + 2])
        _plot_metric(ax, name, results[name], bins, thresholds.get(name), xmax)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)


def analyze_and_plot(
    hists: dict[str, np.ndarray],
    metric_names: list[str],
    out_path,
    title: str,
    *,
    max_nodes_per_object: int = 300,
    n_queries: int = 400,
    group_regex: str | None = None,
    bins: int = 60,
    seed: int = 42,
    xmax: float | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Run the full distance analysis for ``hists`` and write the plot.

    Subsamples, collects within/across (and optional group) distances for each
    metric, prints per-regime stats + the best discriminating threshold, and saves
    a multi-metric figure to ``out_path``. ``xmax`` optionally caps the plot
    x-axis. Returns the raw distance arrays.
    """
    rng = np.random.default_rng(seed)
    group_of = build_groups(list(hists), group_regex)
    if group_of is not None:
        print("Shape groups:")
        for grp in sorted(set(group_of.values())):
            members = [o for o, g in group_of.items() if g == grp]
            print(f"  {grp}: {members}")

    hists = subsample(hists, max_nodes_per_object, rng)
    results = {
        name: collect_distances(hists, METRICS[name], rng, n_queries, group_of)
        for name in metric_names
    }
    thresholds = {name: summarize(name, results[name]) for name in metric_names}
    plot_distance_distributions(results, thresholds, out_path, title, bins, xmax)
    print(f"\nWrote plot to {out_path}")
    return results
