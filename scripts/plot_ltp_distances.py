#!/usr/bin/env python
"""Plot the distribution of LTP (texture-histogram) distances for a dataset.

A Monty "dataset" here is a pretrained ``model.pt`` whose object graphs store an
``ltp`` (Local Ternary Pattern) histogram per node. This script extracts those
histograms and plots how far apart they are under a chosen distance metric,
split into two regimes:

    within-object  : pairs of nodes that belong to the *same* object
    across-object  : pairs of nodes that belong to *different* objects

A good texture metric is one where the within-object distances are small and the
across-object distances are large (little overlap), so this plot is the tool for
calibrating the ``ltp`` tolerance/weight (see
``LTP_BRANCH_COMPARISON_ANALYSIS.md`` section 4b).

For each metric the script sweeps the tolerance and prints the feature evidence
Monty would actually accumulate at each one -- the mean evidence a same-object
node earns, the mean a different-object node earns, and the gap between them --
then recommends the tolerance that maximizes that gap. The tolerance is *not* a
distance threshold: Monty maps the distance through the ramp
``max(0, 1 - d/tol)`` and adds it to the pose evidence, so what matters is the
evidence gap, not how well a hard cutoff classifies pairs. A Youden's-J cutoff is
still printed for reference, but it lands well below the gap-maximizing tolerance
and clips a large share of same-object nodes to zero evidence.

The metric is **not** hard-coded: pass ``--metric`` to pick one (or ``all`` to
compare every registered metric side by side). The ``chisq`` and ``hellinger``
metrics are Monty's actual matching functions (``DefaultFeatureEvidenceCalculator``
/ ``HellingerFeatureEvidenceCalculator``), so their distances are exactly what
Monty computes; ``l1``, ``l2``, ``intersection``, ``bhattacharyya``,
``correlation``, ``kldiv`` and ``jensen_shannon`` are extras for comparison.
Metrics live in
``tbp.monty.frameworks.utils.ltp_distance_analysis.METRICS``; add your own there.

To capture the same distances from a live Monty model via the experiment system
instead, run ``python run.py experiment=plot_ltp_distances`` (see
``MontyLTPDistanceExperiment``).

Usage::

    conda activate tbp.monty
    # default model + chi-square
    python scripts/plot_ltp_distances.py
    # explicit model, Hellinger distance
    python scripts/plot_ltp_distances.py --model <dir-with-model.pt> --metric hellinger
    # compare every metric in one figure
    python scripts/plot_ltp_distances.py --metric all
    # add a third "across-shape-group" regime by grouping objects on a regex
    # (e.g. treat the reskin suffix family as one shape group):
    python scripts/plot_ltp_distances.py --group-regex '(baseball|softball)$'
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tbp.monty.frameworks.utils.ltp_distance_analysis import (
    METRICS,
    analyze_and_plot,
    histograms_from_model_file,
)

DEFAULT_MODEL = (
    "~/tbp/results/monty/pretrained_models/my_trained_models/"
    "supervised_pre_training_11retextured_obj_ltp/pretrained"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to a pretrained run dir (containing model.pt) or to model.pt itself.",
    )
    parser.add_argument(
        "--metric",
        default="chisq",
        help=(
            f"Distance metric: one of {sorted(METRICS)}, 'all', or a "
            "comma-separated subset (e.g. 'chisq,hellinger')."
        ),
    )
    parser.add_argument(
        "--feature",
        default="ltp",
        help="Name of the histogram feature in the model (default: ltp).",
    )
    parser.add_argument("--lm-id", type=int, default=0, help="Learning-module id.")
    parser.add_argument(
        "--max-nodes-per-object",
        type=int,
        default=300,
        help="Subsample each object to at most this many nodes.",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=400,
        help="Number of query nodes to draw distances from.",
    )
    parser.add_argument(
        "--group-regex",
        default=None,
        help=(
            "Optional regex; the matched substring of each object id defines its "
            "shape group. Adds within-group/across-group regimes (e.g. to isolate "
            "same-shape reskins). Objects with no match form their own group."
        ),
    )
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins in plot.")
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help=(
            "Upper bound for the plot x-axis (applied to every metric). "
            "Default: auto-scale each subplot to its 99.5th percentile."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: ltp_distances_<metric>.png in cwd).",
    )
    args = parser.parse_args()

    if args.metric == "all":
        selected_metrics = sorted(METRICS)
    else:
        selected_metrics = [m.strip() for m in args.metric.split(",") if m.strip()]
        unknown = [m for m in selected_metrics if m not in METRICS]
        if unknown:
            parser.error(
                f"Unknown metric(s) {unknown}; choose from {sorted(METRICS)} or 'all'."
            )

    model_dir = Path(args.model).expanduser()
    hists = histograms_from_model_file(model_dir, args.feature, args.lm_id)
    n_bins = next(iter(hists.values())).shape[1]
    print(
        f"Loaded {len(hists)} objects, {sum(h.shape[0] for h in hists.values())} "
        f"nodes, feature {args.feature!r} with {n_bins} bins."
    )

    default_stem = "all" if args.metric == "all" else "_".join(selected_metrics)
    out = Path(args.out or f"ltp_distances_{default_stem}.png").expanduser()

    analyze_and_plot(
        hists,
        selected_metrics,
        out_path=out,
        title=f"LTP ({args.feature}) distances — {model_dir.name}",
        max_nodes_per_object=args.max_nodes_per_object,
        n_queries=args.n_queries,
        group_regex=args.group_regex,
        bins=args.bins,
        seed=args.seed,
        xmax=args.xmax,
    )


if __name__ == "__main__":
    main()
