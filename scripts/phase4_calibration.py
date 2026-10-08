# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
r"""Phase 4: calibrate each texture feature's tolerance on training graphs.

Same rule for every feature (JOURNAL_EXPERIMENT_PLAN.md, Phase 4): choose the
tolerance that maximizes mean evidence(same object) - mean evidence(other
object) under the evidence ramp Monty applies, ``clip(tol - d, 0) / tol``.
Distances are the ones Monty's ``DefaultFeatureEvidenceCalculator`` computes:

- ``ltp``, ``ltp_rgb``, ``ltp_hsv``: Hellinger distance between histograms;
- ``gabor``, ``gabor_rgb``, ``gabor_hsv``: Euclidean distance between the
  stored vectors divided by sqrt(number of channels).

Each model is calibrated on its own graphs (one per dataset x lighting). Two
"other object" regimes are reported:

- ``across_object``: every other object -- the plan's rule;
- ``same_shape``: other variants of the same shape (only where shapes repeat,
  i.e. 3DC-Texture) -- the confusers only texture or color can separate, as in
  the Phase 2 separability measure.

Nodes are subsampled per object (``--max-nodes``) to keep the pairwise
distance sample in memory, and each regime's distances to ``--max-pairs``
before the tolerance sweep (200 tolerances x every pair is otherwise the
bottleneck; the means it estimates are stable well below 1M pairs).

Usage::

    python scripts/phase4_calibration.py \
        --model texture_amb=<pretrained dir> --model mixed_amb=<pretrained dir> \
        --out docs/ltp/results/phase4/tolerances.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
)
from tbp.monty.frameworks.utils.ltp_distance_analysis import (
    METRICS,
    best_tolerance,
    collect_distances,
    extract_histograms,
    subsample,
)

FEATURES = ["ltp", "ltp_rgb", "ltp_hsv", "gabor", "gabor_rgb", "gabor_hsv"]
#: `table_42e_c0_wood_00` -> shape key `table_42e`.
SHAPE_REGEX = r"^(.+?_[0-9a-f]{3})_c\d+_"
VECTOR_CHANNELS = DefaultFeatureEvidenceCalculator.VECTOR_FEATURE_CHANNELS


def distance_for(feature: str):
    """Monty's distance for ``feature``.

    Returns:
        A function ``(stored (n, d), query (d,)) -> distances (n,)``.
    """
    if feature in VECTOR_CHANNELS:
        root_channels = np.sqrt(VECTOR_CHANNELS[feature])
        return (
            lambda stored, query: np.linalg.norm(stored - query, axis=1) / root_channels
        )
    # Equals Monty's Hellinger (cv2 HISTCMP_BHATTACHARYYA) on normalized
    # histograms; see scripts/phase2_footprint_analysis.py.
    return METRICS["bhattacharyya"]


def load_graph_memory(model_dir: str) -> dict:
    path = Path(model_dir).expanduser()
    path = path if path.name == "model.pt" else path / "model.pt"
    state = torch.load(path, weights_only=False)
    return state["lm_dict"][0]["graph_memory"]


def calibrate(
    name: str,
    models: dict,
    feature: str,
    max_nodes: int,
    n_queries: int,
    max_pairs: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    values = extract_histograms(
        models, feature, normalize=feature not in VECTOR_CHANNELS
    )
    values = subsample(values, max_nodes, rng)
    pattern = re.compile(SHAPE_REGEX)
    group_of = {o: (m.group(1) if (m := pattern.match(o)) else o) for o in values}
    dists = collect_distances(
        values, distance_for(feature), rng, n_queries=n_queries, group_of=group_of
    )
    dists = {
        k: (rng.choice(v, max_pairs, replace=False) if v.size > max_pairs else v)
        for k, v in dists.items()
    }
    row = {
        "model": name,
        "feature": feature,
        "n_objects": len(values),
        "n_shapes": len(set(group_of.values())),
        "nodes_used": sum(v.shape[0] for v in values.values()),
        "median_d_within_object": float(np.median(dists["within_object"])),
        "median_d_across_object": float(np.median(dists["across_object"])),
    }
    for regime, across in (
        ("across_object", dists["across_object"]),
        ("same_shape", dists["within_group"]),
    ):
        if across.size == 0:  # no shape repeats (3DC-Mixed)
            continue
        best = best_tolerance(dists["within_object"], across)
        if best is None:
            continue
        top = best["by_gap"]
        row.update(
            {
                f"{regime}_tolerance": top["tolerance"],
                f"{regime}_gap": top["gap"],
                f"{regime}_ev_within": top["ev_within"],
                f"{regime}_ev_across": top["ev_across"],
                f"{regime}_frac_within_clipped": top["frac_within_clipped"],
                f"{regime}_d_prime_at_tolerance": next(
                    r["d_prime"]
                    for r in best["sweep"]
                    if r["tolerance"] == top["tolerance"]
                ),
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="<name>=<pretrained dir or model.pt>; repeat per dataset x lighting",
    )
    parser.add_argument("--features", nargs="+", default=FEATURES)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--max-pairs", type=int, default=2_000_000)
    parser.add_argument("--n-queries", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for spec in args.model:
        name, model_dir = spec.split("=", 1)
        models = load_graph_memory(model_dir)
        for feature in args.features:
            rows.append(
                calibrate(
                    name,
                    models,
                    feature,
                    args.max_nodes,
                    args.n_queries,
                    args.max_pairs,
                    args.seed,
                )
            )
            row = rows[-1]
            print(
                f"{name:>14} {feature:>10}  tol={row['across_object_tolerance']:.3f}"
                f"  gap={row['across_object_gap']:.3f}",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
