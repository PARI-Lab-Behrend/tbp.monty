# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Phase 2 (scale / footprint sweep) offline measures from pretrained graphs.

For each `target_size` run, from the stored `ltp_rgb` histograms only:

1. Object separability against same-shape confusers: Hellinger distances between
   nodes of the same object vs nodes of a *different variant of the same shape*,
   scored by the evidence gap Monty would see at the gap-optimal tolerance
   (`best_tolerance`) and by d'.
2. Location specificity: per object, AUC that spatially far node pairs differ
   more in `ltp_rgb` than near pairs (0.5 = no location information). Reported
   at absolute thresholds (near < 1 cm, far > 5 cm; what Monty's matching sees,
   max_match_distance = 1 cm) and at size-relative ones (the same fractions of
   the object as 1 / 5 cm are of a 0.12 m object).

Decision rule (JOURNAL_EXPERIMENT_PLAN.md, Phase 2): the footprint with the
largest separability subject to absolute location-specificity AUC > 0.6.

Usage:
    python scripts/phase2_footprint_analysis.py \
        --run 0.12=<model dir> --run 0.25=<model dir> --run 0.5=<model dir> \
        --out phase2_results.csv
"""

from __future__ import annotations

import argparse
import csv
import re

import numpy as np

from tbp.monty.frameworks.utils.ltp_distance_analysis import (
    METRICS,
    best_tolerance,
    collect_distances,
    extract_histograms,
    extract_positions,
    location_specificity_auc,
)

#: `table_46b_c0_wood_13` -> shape key `table_46b`.
DEFAULT_SHAPE_REGEX = r"^(.+?_[0-9a-f]{3})_c\d+_"


def load_models(model_dir):
    from pathlib import Path

    import torch

    path = Path(model_dir).expanduser()
    path = path if path.name == "model.pt" else path / "model.pt"
    state = torch.load(path, weights_only=False)
    return state["lm_dict"][0]["graph_memory"]


def analyze(
    size: float, model_dir: str, feature: str, seed: int, shape_regex: str
) -> dict:
    rng = np.random.default_rng(seed)
    models = load_models(model_dir)
    hists = extract_histograms(models, feature)
    positions = extract_positions(models)
    pattern = re.compile(shape_regex)
    group_of = {
        o: (m.group(1) if (m := pattern.match(o)) else o) for o in hists
    }
    # Monty's Hellinger (cv2 HISTCMP_BHATTACHARYYA) equals the vectorized
    # `bhattacharyya` metric on normalized histograms (max diff ~2e-9), and is
    # ~50x faster than the per-node cv2 loop.
    hellinger = METRICS["bhattacharyya"]

    dists = collect_distances(hists, hellinger, rng, n_queries=2000, group_of=group_of)
    sep = best_tolerance(dists["within_object"], dists["within_group"])

    scale = size / 0.12
    aucs_abs, aucs_rel = [], []
    per_shape: dict[str, list[float]] = {}
    for obj in hists:
        a = location_specificity_auc(positions[obj], hists[obj], hellinger, rng)
        r = location_specificity_auc(
            positions[obj], hists[obj], hellinger, rng,
            near=0.01 * scale, far=0.05 * scale,
        )
        aucs_abs.append(a["auc"])
        aucs_rel.append(r["auc"])
        per_shape.setdefault(group_of[obj], []).append(a["auc"])

    return {
        "target_size": size,
        "n_objects": len(hists),
        "median_nodes_per_object": int(np.median([h.shape[0] for h in hists.values()])),
        "median_d_within_object": float(np.median(dists["within_object"])),
        "median_d_same_shape_other_variant": float(np.median(dists["within_group"])),
        "best_tolerance": sep["by_gap"]["tolerance"],
        "evidence_gap": sep["by_gap"]["gap"],
        "d_prime": sep["by_d_prime"]["d_prime"],
        "loc_auc_abs_median": float(np.nanmedian(aucs_abs)),
        "loc_auc_abs_min": float(np.nanmin(aucs_abs)),
        "loc_auc_rel_median": float(np.nanmedian(aucs_rel)),
        **{f"loc_auc_abs_{s}": float(np.nanmedian(v)) for s, v in sorted(per_shape.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True,
                        help="<target_size>=<model dir or model.pt>; repeat per size")
    parser.add_argument("--feature", default="ltp_rgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shape-regex", default=DEFAULT_SHAPE_REGEX,
                        help="Regex whose group 1 is an object's shape key")
    parser.add_argument("--out", default="phase2_results.csv")
    args = parser.parse_args()

    rows = []
    for spec in args.run:
        size, model_dir = spec.split("=", 1)
        rows.append(analyze(float(size), model_dir, args.feature, args.seed,
                            args.shape_regex))

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    width = max(len(k) for k in keys)
    print(f"{'':<{width}}" + "".join(f"{r['target_size']:>12}" for r in rows))
    for k in keys[1:]:
        cells = "".join(
            f"{r.get(k, float('nan')):>12.4f}" if isinstance(r.get(k), float)
            else f"{r.get(k, ''):>12}" for r in rows
        )
        print(f"{k:<{width}}{cells}")

    ok = [r for r in rows if r["loc_auc_abs_median"] > 0.6]
    if ok:
        best = max(ok, key=lambda r: r["evidence_gap"])
        print(f"\nDecision rule -> target_size {best['target_size']} "
              f"(max evidence gap among sizes with location AUC > 0.6)")
    else:
        print("\nDecision rule: no size has location AUC > 0.6 (see plan's fallback)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
