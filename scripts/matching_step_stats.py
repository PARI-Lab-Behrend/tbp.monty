#!/usr/bin/env python
"""Report mean / std / SEM of matching steps per Monty eval run.

Each Monty eval run writes an ``eval_stats.csv`` (one row per episode). The
matching-step count lives in ``monty_matching_steps`` (identical to
``num_steps``). This gives you the error bars the summary logs omit.

Usage::

    # by experiment/run name (searched under the results root)
    python scripts/matching_step_stats.py <run_name> [<run_name> ...]

    # or by explicit path to a run dir or an eval_stats.csv
    python scripts/matching_step_stats.py /path/to/run_dir

    # restrict to correctly-classified episodes only
    python scripts/matching_step_stats.py <run_name> --correct-only

Set MONTY_RESULTS_ROOT to change where names are searched
(default: ~/tbp/results/monty).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

STEP_COL = "monty_matching_steps"
DEFAULT_ROOT = Path(
    os.environ.get("MONTY_RESULTS_ROOT", "~/tbp/results/monty")
).expanduser()


def resolve_csv(token: str, root: Path) -> Path | None:
    """Resolve a run name or path to an eval_stats.csv."""
    p = Path(token).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        cand = p / "eval_stats.csv"
        if cand.is_file():
            return cand
    # Treat as a run name: find a run dir with that name under root.
    matches = sorted(
        c for c in root.rglob("eval_stats.csv") if c.parent.name == token
    )
    if matches:
        if len(matches) > 1:
            print(f"WARNING: multiple runs named '{token}', using {matches[0].parent}")
        return matches[0]
    return None


def stats(csv_path: Path, correct_only: bool) -> dict:
    df = pd.read_csv(csv_path)
    if correct_only and "primary_performance" in df:
        df = df[df["primary_performance"].astype(str).str.startswith("correct")]
    s = pd.to_numeric(df[STEP_COL], errors="coerce").dropna()
    n = len(s)
    return {
        "n": n,
        "mean": s.mean(),
        "std": s.std(),  # sample std (ddof=1)
        "sem": s.std() / n**0.5 if n else float("nan"),
        "median": s.median(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run names or paths")
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Only count episodes with primary_performance starting with 'correct'",
    )
    args = parser.parse_args()

    rows = []
    for token in args.runs:
        csv_path = resolve_csv(token, DEFAULT_ROOT)
        if csv_path is None:
            print(f"WARNING: no eval_stats.csv found for '{token}'")
            continue
        s = stats(csv_path, args.correct_only)
        rows.append({"run": csv_path.parent.name, **s})

    if not rows:
        print("Nothing to report.")
        return

    table = pd.DataFrame(rows)
    for c in ("mean", "std", "sem", "median"):
        table[c] = table[c].round(2)
    scope = "correct episodes only" if args.correct_only else "all episodes"
    print(f"matching steps ({STEP_COL}), {scope}:\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
