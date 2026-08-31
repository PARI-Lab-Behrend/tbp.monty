#!/usr/bin/env python
"""Paired statistical comparison of two Monty eval runs.

Given two eval runs (each an ``eval_stats.csv`` or a directory containing one),
align them episode-by-episode and test whether they differ on a chosen metric and
on recognition accuracy. Designed for LTP-vs-no-LTP style comparisons where both
arms are evaluated on the *same* objects and random rotations (seeded), so episodes
pair up one-to-one.

Why paired: the arms share the same object x rotation conditions, and per-condition
difficulty dominates the metric variance. Pairing removes that shared variance and
is far more powerful than an unpaired test when the paired correlation is high
(which it is for these runs). Both paired and unpaired results are reported so the
gain is visible.

Tests reported (no SciPy required; SciPy used for Wilcoxon/Mann-Whitney if present):
    metric  : paired permutation (sign-flip) + unpaired permutation, means/medians,
              paired correlation, effect size, and Wilcoxon/Mann-Whitney if SciPy.
    accuracy: exact McNemar (paired) on primary_performance.startswith("correct").
    convergence (censoring-aware): fraction of episodes below --max-steps, plus
              exact McNemar on convergence -- the meaningful signal when many
              episodes hit the step cap.

Usage::

    python scripts/pvalue_test.py A B [options]
    # A, B: eval_stats.csv paths OR run directories (recursively finds eval_stats.csv)
    # Recovered snapshots (eval_stats_*.csv) are accepted as explicit file paths.

    python scripts/pvalue_test.py \\
        ~/tbp/results/monty/projects/monty_runs/pval_amb_11retex_surf_agent_ltp \\
        ~/tbp/results/monty/projects/monty_runs/pval_amb_11retex_surf_agent_noltp \\
        --label-a LTP --label-b noLTP --metric monty_matching_steps
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
from pathlib import Path

import pandas as pd

try:  # SciPy is optional; permutation/exact tests cover the same ground without it.
    from scipy.stats import mannwhitneyu, wilcoxon

    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False


def resolve_csv(source: str) -> Path:
    """Return the eval_stats.csv for a source that may be a file or a directory."""
    p = Path(source).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        # Prefer a top-level eval_stats.csv, else the first found recursively.
        top = p / "eval_stats.csv"
        if top.is_file():
            return top
        found = sorted(p.rglob("eval_stats.csv"))
        if found:
            return found[0]
    raise FileNotFoundError(f"No eval_stats.csv found for source: {source}")


def load(source: str, key_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(resolve_csv(source))
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{source}: missing key column(s) {missing}")
    return df


def align(a: pd.DataFrame, b: pd.DataFrame, key_cols: list[str]):
    """Inner-join two eval frames on the alignment key; report drops."""
    a = a.copy()
    b = b.copy()
    a["_key"] = a[key_cols].astype(str).agg("|".join, axis=1)
    b["_key"] = b[key_cols].astype(str).agg("|".join, axis=1)
    # Guard against accidental duplicate keys (would silently multiply rows).
    a = a.drop_duplicates("_key")
    b = b.drop_duplicates("_key")
    merged = a.merge(b, on="_key", suffixes=("_a", "_b"))
    return merged, len(a), len(b)


def two_sided_perm_paired(diff: list[float], n_perm: int, seed: int) -> float:
    """Sign-flip permutation p-value for mean(diff) == 0 (assumption-light)."""
    diff = [d for d in diff if d is not None and not math.isnan(d)]
    if not diff:
        return float("nan")
    obs = abs(st.mean(diff))
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        m = st.mean(d if rng.random() < 0.5 else -d for d in diff)
        if abs(m) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def two_sided_perm_unpaired(a: list[float], b: list[float], n_perm: int, seed: int) -> float:
    """Label-shuffle permutation p-value for mean(a) == mean(b)."""
    a = [x for x in a if not math.isnan(x)]
    b = [x for x in b if not math.isnan(x)]
    pool = a + b
    n = len(a)
    obs = abs(st.mean(a) - st.mean(b))
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(st.mean(pool[:n]) - st.mean(pool[n:])) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def exact_mcnemar(b10: int, b01: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b10 + b01
    if n == 0:
        return 1.0
    k = min(b10, b01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = st.mean(x), st.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))
    return num / den if den else float("nan")


def is_correct(series: pd.Series) -> pd.Series:
    return series.astype(str).str.startswith("correct")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="First run: eval_stats.csv or run directory")
    ap.add_argument("b", help="Second run: eval_stats.csv or run directory")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--metric", default="monty_matching_steps",
                    help="Numeric column to test (default: monty_matching_steps)")
    ap.add_argument("--key", default="primary_target_object,episode_seed",
                    help="Comma-separated columns to align episodes on")
    ap.add_argument("--max-steps", type=float, default=500,
                    help="Step cap; episodes at/above this are 'not converged' "
                         "for the censoring-aware convergence test")
    ap.add_argument("--n-perm", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", help="Optional path to write a one-row summary CSV")
    args = ap.parse_args()

    key_cols = [c.strip() for c in args.key.split(",") if c.strip()]
    la, lb = args.label_a, args.label_b

    da = load(args.a, key_cols)
    db = load(args.b, key_cols)
    merged, na, nb = align(da, db, key_cols)
    n = len(merged)

    print("=" * 70)
    print(f"{la}: {resolve_csv(args.a)}")
    print(f"{lb}: {resolve_csv(args.b)}")
    print(f"aligned pairs: {n}   ({la} rows={na}, {lb} rows={nb}, key={key_cols})")
    if n == 0:
        print("No aligned episodes -- check the --key columns match between runs.")
        return
    if n < min(na, nb):
        print(f"WARNING: {min(na, nb) - n} episode(s) did not align and were dropped.")

    summary: dict[str, object] = {"label_a": la, "label_b": lb, "n_pairs": n,
                                  "metric": args.metric}

    # ---- ACCURACY (paired McNemar) ----
    ca = is_correct(merged["primary_performance_a"])
    cb = is_correct(merged["primary_performance_b"])
    acc_a, acc_b = ca.mean(), cb.mean()
    a_right_b_wrong = int((ca & ~cb).sum())
    b_right_a_wrong = int((cb & ~ca).sum())
    p_acc = exact_mcnemar(a_right_b_wrong, b_right_a_wrong)
    print("\n--- ACCURACY (correct incl. mlh) ---")
    print(f"{la}={acc_a*100:.1f}%   {lb}={acc_b*100:.1f}%")
    print(f"discordant: {la}-right/{lb}-wrong={a_right_b_wrong}, "
          f"{lb}-right/{la}-wrong={b_right_a_wrong}")
    print(f"exact McNemar p (2-sided) = {p_acc:.4f}")
    summary.update(acc_a=acc_a, acc_b=acc_b, mcnemar_p=p_acc)

    # ---- METRIC (paired + unpaired) ----
    ma = args.metric + "_a"
    mb = args.metric + "_b"
    if ma in merged and mb in merged:
        va = pd.to_numeric(merged[ma], errors="coerce")
        vb = pd.to_numeric(merged[mb], errors="coerce")
        mask = va.notna() & vb.notna()
        av = va[mask].tolist()
        bv = vb[mask].tolist()
        diff = [x - y for x, y in zip(av, bv)]  # A - B
        r = pearson(av, bv)
        mean_a, mean_b = st.mean(av), st.mean(bv)
        p_paired = two_sided_perm_paired(diff, args.n_perm, args.seed)
        p_unpaired = two_sided_perm_unpaired(av, bv, args.n_perm, args.seed)
        dz = st.mean(diff) / st.pstdev(diff) if st.pstdev(diff) else float("nan")
        print(f"\n--- METRIC: {args.metric} ---")
        print(f"{la}: mean={mean_a:.2f} median={st.median(av):.1f}")
        print(f"{lb}: mean={mean_b:.2f} median={st.median(bv):.1f}")
        print(f"mean diff ({la}-{lb})={st.mean(diff):.2f}   Cohen's dz={dz:.3f}")
        print(f"paired correlation r={r:.3f}   "
              f"SD indiv={st.pstdev(av + bv):.2f}   SD diff={st.pstdev(diff):.2f}")
        print(f"PAIRED   permutation p (2-sided) = {p_paired:.4f}")
        print(f"UNPAIRED permutation p (2-sided) = {p_unpaired:.4f}")
        if _HAVE_SCIPY:
            try:
                w_p = wilcoxon(av, bv).pvalue
                print(f"Wilcoxon signed-rank p (2-sided) = {w_p:.4f}")
                summary["wilcoxon_p"] = w_p
            except ValueError as exc:  # e.g. all-zero differences
                print(f"Wilcoxon: {exc}")
            mw_p = mannwhitneyu(av, bv, alternative="two-sided").pvalue
            print(f"Mann-Whitney U p (2-sided) = {mw_p:.4f}")
            summary["mannwhitney_p"] = mw_p
        else:
            print("(install SciPy for Wilcoxon signed-rank / Mann-Whitney)")
        summary.update(metric_mean_a=mean_a, metric_mean_b=mean_b,
                       metric_mean_diff=st.mean(diff), paired_r=r,
                       paired_perm_p=p_paired, unpaired_perm_p=p_unpaired)

        # ---- CONVERGENCE (censoring-aware) ----
        conv_a = va < args.max_steps
        conv_b = vb < args.max_steps
        cav = conv_a[mask]
        cbv = conv_b[mask]
        a_conv_b_not = int((cav & ~cbv).sum())
        b_conv_a_not = int((cbv & ~cav).sum())
        p_conv = exact_mcnemar(a_conv_b_not, b_conv_a_not)
        print(f"\n--- CONVERGENCE (< {args.max_steps:.0f} steps) ---")
        print(f"{la}={cav.mean()*100:.1f}% converged   {lb}={cbv.mean()*100:.1f}% converged")
        print(f"discordant: {la}-conv/{lb}-not={a_conv_b_not}, "
              f"{lb}-conv/{la}-not={b_conv_a_not}")
        print(f"exact McNemar p (2-sided) = {p_conv:.4f}")
        summary.update(conv_rate_a=float(cav.mean()), conv_rate_b=float(cbv.mean()),
                       conv_mcnemar_p=p_conv)
    else:
        print(f"\nWARNING: metric column '{args.metric}' not found in both runs; "
              "skipping metric/convergence tests.")

    if args.csv:
        out = Path(args.csv).expanduser()
        pd.DataFrame([summary]).to_csv(out, index=False)
        print(f"\nWrote summary to {out}")


if __name__ == "__main__":
    main()
