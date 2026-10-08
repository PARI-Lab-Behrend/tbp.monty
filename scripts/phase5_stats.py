# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
r"""Phase 5 statistics: per-condition summary, planned comparisons, interaction.

Reads the eval runs of one dataset (``eval_3dc_<dataset>_<lighting>_<condition>``,
optionally with a ``_seed<N>`` suffix per extra seed; all seed runs of a
condition are pooled) under ``--root`` and writes, for each lighting found:

1. ``summary_<dataset>_<lighting>.csv``: per condition, episodes, percent
   correct (``correct`` + ``correct_mlh``, as Monty's ``percent_correct``),
   percent timed out (``*_mlh``: the most-likely hypothesis was used after the
   step cap), and mean +- SEM matching steps (``monty_matching_steps``).
2. ``comparisons_<dataset>_<lighting>.csv``: the planned comparisons
   (proposed vs baseline, vs ``ltp``, vs Gabor matched), paired by
   (object, episode seed): exact McNemar on correctness, Wilcoxon signed-rank on
   steps, Holm-corrected within each test family, bootstrap 95% CIs on the
   accuracy and mean-steps differences.
3. ``curve_<dataset>_<lighting>.csv``: steps-to-correct curve per condition --
   the fraction of episodes recognized (``correct``, i.e. converged before the
   cap) by matching step k.
4. With both lightings: ``interaction_<dataset>.csv``: (proposed - baseline)
   under directional minus under ambient, on accuracy and mean steps, with a
   bootstrap 95% CI resampling (object, episode seed) units present in all
   four runs.

Condition names are the eval-config suffixes from
``scripts/generate_journal_configs.py``. ``--color-ltp`` names the Phase 3
winner; proposed is ``hsv_plus_<winner>`` and Gabor matched is
``hsv_plus_gabor_<rgb|hsv>``.

Usage::

    python scripts/phase5_stats.py --root ~/tbp/results/monty/projects/journal \\
        --dataset texture --color-ltp ltp_rgb --out docs/ltp/results/phase5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvalue_test import exact_mcnemar, is_correct, resolve_csv

KEY = ["primary_target_object", "episode_seed"]
STEPS = "monty_matching_steps"
LIGHTINGS = ("amb", "dir")


def planned(color_ltp: str) -> dict[str, tuple[str, str]]:
    """Planned comparisons for a Phase 3 winner.

    Returns:
        Comparison name -> (proposed condition, comparison condition).
    """
    proposed = f"hsv_plus_{color_ltp}"
    gabor = "gabor_" + color_ltp[len("ltp_") :]
    return {
        "proposed_vs_baseline": (proposed, "hsv"),
        "proposed_vs_ltp": (proposed, "ltp"),
        "proposed_vs_gabor_matched": (proposed, f"hsv_plus_{gabor}"),
    }


def find_runs(root: Path, dataset: str, lighting: str) -> dict[str, pd.DataFrame]:
    """Load every condition's eval runs for one dataset and lighting.

    Returns:
        Condition -> pooled eval rows (LM_0, all seed runs), indexed by KEY.

    Raises:
        ValueError: if a condition has two rows for one (object, episode seed).
    """
    pattern = re.compile(rf"^eval_3dc_{dataset}_{lighting}_(.+?)(?:_seed\d+)?$")
    frames: dict[str, list[pd.DataFrame]] = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        match = pattern.match(run_dir.name)
        if not match:
            continue
        df = pd.read_csv(resolve_csv(str(run_dir)))
        if "lm_id" in df.columns:
            df = df[df["lm_id"] == "LM_0"]
        df["run"] = run_dir.name
        frames.setdefault(match.group(1), []).append(df)
    runs = {}
    for condition, dfs in frames.items():
        df = pd.concat(dfs, ignore_index=True)
        duplicated = df.duplicated(KEY)
        if duplicated.any():
            raise ValueError(
                f"{condition}: {duplicated.sum()} duplicate (object, episode_seed) rows"
            )
        df["correct"] = is_correct(df["primary_performance"])
        df["timed_out"] = df["primary_performance"].astype(str).str.endswith("_mlh")
        df["converged"] = df["primary_performance"].astype(str) == "correct"
        runs[condition] = df.set_index(KEY)
    return runs


def summarize(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for condition, df in sorted(runs.items()):
        steps = df[STEPS].astype(float)
        rows.append(
            {
                "condition": condition,
                "episodes": len(df),
                "percent_correct": 100 * df["correct"].mean(),
                "percent_timed_out": 100 * df["timed_out"].mean(),
                "mean_steps": steps.mean(),
                "sem_steps": steps.std(ddof=1) / np.sqrt(len(steps)),
            }
        )
    return pd.DataFrame(rows)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down correction.

    Returns:
        Adjusted p-values, in the input order.
    """
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_ci(
    values: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Bootstrap over paired units (the rows of ``values``).

    Returns:
        95% percentile CI of the mean.
    """
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ci(name: str, values: np.ndarray, n_boot: int, rng) -> dict[str, float]:
    """Bootstrap CI as two columns.

    Returns:
        ``{<name>_ci95_lo, <name>_ci95_hi}``.
    """
    lo, hi = bootstrap_ci(values, n_boot, rng)
    return {f"{name}_ci95_lo": lo, f"{name}_ci95_hi": hi}


def paired(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def compare(
    runs: dict[str, pd.DataFrame], color_ltp: str, n_boot: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, (cond_a, cond_b) in planned(color_ltp).items():
        if cond_a not in runs or cond_b not in runs:
            print(f"skip {name}: missing {cond_a if cond_a not in runs else cond_b}")
            continue
        a, b = paired(runs[cond_a], runs[cond_b])
        acc_diff = a["correct"].astype(float).to_numpy() - b["correct"].to_numpy()
        steps_diff = a[STEPS].to_numpy(float) - b[STEPS].to_numpy(float)
        only_a = int((a["correct"] & ~b["correct"]).sum())
        only_b = int((~a["correct"] & b["correct"]).sum())
        nonzero = steps_diff[steps_diff != 0]
        rows.append(
            {
                "comparison": name,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "pairs": len(a),
                "unpaired_a": len(runs[cond_a]) - len(a),
                "unpaired_b": len(runs[cond_b]) - len(b),
                "accuracy_diff_points": 100 * acc_diff.mean(),
                **ci("accuracy_diff", 100 * acc_diff, n_boot, rng),
                "only_a_correct": only_a,
                "only_b_correct": only_b,
                "mcnemar_p": exact_mcnemar(only_a, only_b),
                "steps_diff_mean": steps_diff.mean(),
                **ci("steps_diff", steps_diff, n_boot, rng),
                "wilcoxon_p": (
                    float(wilcoxon(nonzero).pvalue) if nonzero.size else 1.0
                ),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["mcnemar_p_holm"] = holm(df["mcnemar_p"].tolist())
        df["wilcoxon_p_holm"] = holm(df["wilcoxon_p"].tolist())
    return df


def curve(runs: dict[str, pd.DataFrame], max_steps: int) -> pd.DataFrame:
    ks = np.arange(0, max_steps + 1)
    out = {"step": ks}
    for condition, df in sorted(runs.items()):
        steps = df.loc[df["converged"], STEPS].to_numpy(float)
        out[condition] = np.searchsorted(np.sort(steps), ks, side="right") / len(df)
    return pd.DataFrame(out)


def interaction(
    by_lighting: dict[str, dict[str, pd.DataFrame]],
    color_ltp: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    proposed, baseline = planned(color_ltp)["proposed_vs_baseline"]
    frames = [
        by_lighting[light][c] for light in LIGHTINGS for c in (proposed, baseline)
    ]
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    amb_p, amb_b, dir_p, dir_b = (f.loc[common] for f in frames)
    rng = np.random.default_rng(seed)
    rows = []
    for metric, column, scale in (
        ("accuracy_points", "correct", 100.0),
        ("mean_steps", STEPS, 1.0),
    ):

        def gap(p, b, column=column, scale=scale):
            return scale * (p[column].to_numpy(float) - b[column].to_numpy(float))

        diff = gap(dir_p, dir_b) - gap(amb_p, amb_b)
        rows.append(
            {
                "metric": metric,
                "units": len(common),
                "gap_amb": gap(amb_p, amb_b).mean(),
                "gap_dir": gap(dir_p, dir_b).mean(),
                "interaction": diff.mean(),
                **ci("interaction", diff, n_boot, rng),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True, type=Path, help="Dir of eval run dirs")
    ap.add_argument("--dataset", required=True, choices=["texture", "mixed"])
    ap.add_argument("--color-ltp", default="ltp_rgb", choices=["ltp_rgb", "ltp_hsv"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    by_lighting = {}
    for lighting in LIGHTINGS:
        runs = find_runs(args.root.expanduser(), args.dataset, lighting)
        if not runs:
            continue
        by_lighting[lighting] = runs
        tag = f"{args.dataset}_{lighting}"
        summary = summarize(runs)
        comparisons = compare(runs, args.color_ltp, args.n_boot, args.seed)
        summary.to_csv(args.out / f"summary_{tag}.csv", index=False)
        comparisons.to_csv(args.out / f"comparisons_{tag}.csv", index=False)
        curve(runs, args.max_steps).to_csv(args.out / f"curve_{tag}.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(f"\n== {tag}\n{summary.round(2).to_string(index=False)}")
            if not comparisons.empty:
                print(comparisons.round(4).to_string(index=False))
    if all(light in by_lighting for light in LIGHTINGS):
        result = interaction(by_lighting, args.color_ltp, args.n_boot, args.seed)
        result.to_csv(args.out / f"interaction_{args.dataset}.csv", index=False)
        print(f"\n== interaction (dir - amb), {args.dataset}\n{result.round(3)}")
    if not by_lighting:
        print(f"No eval_3dc_{args.dataset}_* runs under {args.root}")


if __name__ == "__main__":
    main()
