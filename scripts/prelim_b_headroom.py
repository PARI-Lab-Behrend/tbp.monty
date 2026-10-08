# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Prelim B (headroom): is there room for a texture feature to help?

Reads ``prelim_eval_3dc_<dataset>_amb_<condition>_2rot`` runs under ``--root``
and reports, per dataset x condition: accuracy (``correct`` + ``correct_mlh``),
shape accuracy (the most likely object has the target's shape -- 3DC-Texture's
5 variants share a shape; 3DC-Mixed objects are all distinct shapes, so there
the class is used), percent timed out, and matching steps. Writes a summary
CSV and one confusion matrix (target x most likely object) per run.

Pass criteria (JOURNAL_EXPERIMENT_PLAN.md, Prelim B):

- 3DC-Texture: ``hsv`` accuracy in [30, 80]%; ``shape_only`` near chance among
  the 5 same-shape variants (20%) but correct on shape.
- 3DC-Mixed: ``shape_only`` accuracy > 70%; ``hsv`` not at 100% with no steps
  to spare.

Usage::

    python scripts/prelim_b_headroom.py --root ~/tbp/results/monty/projects/journal \\
        --out docs/ltp/results/prelim_b
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

#: `table_42e_c0_wood_00` -> shape `table_42e`; `bag_057_a8a5231d` -> class `bag`.
SHAPE_REGEX = re.compile(r"^(.+?_[0-9a-f]{3})_c\d+_")
CLASS_REGEX = re.compile(r"^(.+?)_[0-9a-f]{3}_")


def group_key(name: str, dataset: str) -> str:
    match = (SHAPE_REGEX if dataset == "texture" else CLASS_REGEX).match(str(name))
    return match.group(1) if match else str(name)


def summarize(df: pd.DataFrame, dataset: str, condition: str) -> dict:
    performance = df["primary_performance"].astype(str)
    target = df["primary_target_object"].astype(str)
    guess = df["most_likely_object"].astype(str)
    same_group = [
        group_key(t, dataset) == group_key(g, dataset) for t, g in zip(target, guess)
    ]
    steps = df["num_steps"].astype(float)
    return {
        "dataset": dataset,
        "condition": condition,
        "episodes": len(df),
        "percent_correct": 100 * performance.str.startswith("correct").mean(),
        "percent_correct_before_cap": 100 * (performance == "correct").mean(),
        "percent_right_shape_or_class": 100 * pd.Series(same_group).mean(),
        "percent_timed_out": 100 * performance.str.endswith("_mlh").mean(),
        "median_steps": steps.median(),
        "mean_steps": steps.mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", default=["shape_only", "hsv"])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in ("texture", "mixed"):
        for condition in args.conditions:
            run = f"prelim_eval_3dc_{dataset}_amb_{condition}_2rot"
            csv_path = args.root.expanduser() / run / "eval_stats.csv"
            if not csv_path.is_file():
                print(f"missing {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            df = df[df["lm_id"] == "LM_0"] if "lm_id" in df.columns else df
            rows.append(summarize(df, dataset, condition))
            pd.crosstab(df["primary_target_object"], df["most_likely_object"]).to_csv(
                args.out / f"confusion_{dataset}_{condition}.csv"
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summary.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
