#!/usr/bin/env python
"""Compare Monty eval runs by scanning results folders for ``eval_stats.csv``.

Each Monty eval run writes an ``eval_stats.csv`` (one row per episode) into its
run directory. This script walks one or more results folders, finds every such
file, and prints a side-by-side accuracy comparison.

Usage::

    python scripts/compare_eval_runs.py <results_dir> [<results_dir> ...]
    python scripts/compare_eval_runs.py <results_dir> --csv out.csv

Metrics per run:
    episodes    number of episodes (rows)
    strict_acc  fraction with primary_performance == "correct" (converged)
    incl_mlh    fraction with primary_performance starting with "correct"
                (also counts "correct_mlh", i.e. right object as most-likely
                hypothesis at timeout)
    mean_steps  mean num_steps
    mean_rot_err mean rotation_error over episodes where it is defined
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def find_eval_runs(roots: list[str]) -> list[tuple[str, Path]]:
    """Find (run_name, csv_path) for every eval_stats.csv under the given roots.

    run_name is the directory containing the csv, made unique by prefixing the
    parent folder when the same dir name appears under multiple roots.
    """
    found: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for root in roots:
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            print(f"WARNING: not a directory, skipping: {root_path}")
            continue
        for csv_path in sorted(root_path.rglob("eval_stats.csv")):
            resolved = csv_path.resolve()
            # Roots may overlap (e.g. a parent dir and its child both passed);
            # dedupe so each csv is reported once.
            if resolved in seen:
                continue
            seen.add(resolved)
            run_dir = csv_path.parent
            # Label as "<parent_of_run>/<run_dir>" so runs with the same name
            # under different result folders stay distinguishable.
            label = f"{run_dir.parent.name}/{run_dir.name}"
            found.append((label, csv_path))
    return found


def summarize(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    perf = df["primary_performance"].astype(str)
    correct_incl = perf.str.startswith("correct")
    summary = {
        "episodes": len(df),
        "strict_acc": (perf == "correct").mean(),
        "incl_mlh": correct_incl.mean(),
        "mean_steps": df["num_steps"].mean() if "num_steps" in df else float("nan"),
    }
    if "rotation_error" in df:
        summary["mean_rot_err"] = pd.to_numeric(
            df["rotation_error"], errors="coerce"
        ).mean()
    else:
        summary["mean_rot_err"] = float("nan")
    # Keep raw category counts for context.
    summary["_perf_counts"] = perf.value_counts().to_dict()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        help="One or more results folders to scan recursively for eval_stats.csv",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Optional path to also write the comparison table as CSV.",
    )
    args = parser.parse_args()

    runs = find_eval_runs(args.roots)
    if not runs:
        print("No eval_stats.csv files found under the given folders.")
        return

    rows = []
    for label, csv_path in runs:
        try:
            s = summarize(csv_path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"WARNING: failed to read {csv_path}: {exc}")
            continue
        rows.append({"run": label, **s})

    table = pd.DataFrame(rows).sort_values("strict_acc", ascending=False)
    display_cols = [
        "run",
        "episodes",
        "strict_acc",
        "incl_mlh",
        "mean_steps",
        "mean_rot_err",
    ]

    pretty = table[display_cols].copy()
    pretty["strict_acc"] = (pretty["strict_acc"] * 100).round(1)
    pretty["incl_mlh"] = (pretty["incl_mlh"] * 100).round(1)
    pretty["mean_steps"] = pretty["mean_steps"].round(1)
    pretty["mean_rot_err"] = pretty["mean_rot_err"].round(3)
    pretty = pretty.rename(
        columns={"strict_acc": "strict_acc_%", "incl_mlh": "incl_mlh_%"}
    )

    print(pretty.to_string(index=False))

    print("\nPer-run primary_performance breakdown:")
    for label, _ in runs:
        match = table[table["run"] == label]
        if not match.empty:
            print(f"  {label}: {match.iloc[0]['_perf_counts']}")

    if args.csv:
        out = Path(args.csv).expanduser()
        table[display_cols].to_csv(out, index=False)
        print(f"\nWrote comparison table to {out}")


if __name__ == "__main__":
    main()
