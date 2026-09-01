# LBP vs LTP vs control — results (Tier 3, R2's "why not LBP?")

Run 2026-08-20. All numbers from `~/tbp/results/monty/projects/monty_runs_lbpcmp/_snapshots/`.
Reproduce with `scripts/run_lbp_comparison_tuned.sh` (idempotent; skips finished arms).

## Design

Eight eval arms on 11retextured, two lighting conditions, three descriptors:

| arm | descriptor | histogram | tolerance |
|---|---|---|---|
| control | HSV only | — | — |
| LBP | classic LBP, `ror` | 36 bins | 0.50 (inherited from LTP) |
| LBP@0.40 | classic LBP, `ror` | 36 bins | 0.40 (its **own** tuned value) |
| LTP | split LTP, `ror` | 2 x 36 bins | 0.50 |

LBP is `neighbor >= center` with no dead zone and no sign split, and is otherwise
**identical** to the LTP path: same 8 neighbors at radius 1, same bilinear sampling,
same ROR encoding, same masking, same L1 normalization, same feature weight (2.0),
same Hellinger matching, same policy, same seed (42). Implemented as
`local_binary_pattern` in `sensor_processing.py`; reported under the feature name
`ltp` so every downstream LM config applies unchanged.

The LBP arms use their own pretrained models (the stored histogram differs); control
and LTP arms share the existing LTP models, which is sound because the training LM
never uses the texture feature to build graphs.

All arms share objects x rotations by seed, so episodes pair one-to-one:
**n = 220** ambient (20 rotations x 11 objects), **n = 110** directional (10 x 11).
Tests: exact McNemar on accuracy and on convergence, paired permutation + Wilcoxon
on `monty_matching_steps`.

### Replication check

The three arms that overlap the published tables reproduce them to two decimals:
ambient control 93.18 / 218.69, ambient LTP 96.82 / 185.46, directional control
89.09 / 275.83, directional LTP 94.55 / 243.96. The pipeline is the same one that
produced Tables 1-2.

### Tolerance selection (neutralizes the tuning-bias charge, Tier 2)

Gap-maximizing Hellinger tolerance per model, from `scripts/plot_ltp_distances.py`:

| condition | descriptor | recommended tol | max evidence gap |
|---|---|---|---|
| ambient | LTP | 0.481 | 0.487 |
| ambient | LBP | 0.393 | 0.408 |
| directional | LTP | 0.518 | 0.364 |
| directional | LBP | 0.407 | 0.377 |

The paper's 0.50 is confirmed near-optimal **for LTP in both conditions**. LBP's own
optimum is ~0.40, which is why the `@0.40` arms exist: without them the comparison
would be judging LBP at a threshold tuned for a different descriptor. Note LBP's
patch separation is *better* than LTP's under directional light (0.377 vs 0.364) —
LBP is not a strawman at the descriptor level.

## Results

### Ambient, n = 220

| arm | accuracy | steps ± SEM | median | converged <500 |
|---|---|---|---|---|
| control (HSV) | 93.18 | 218.69 ± 15.24 | 49.5 | 61.4% |
| LBP @ 0.50 | 96.82 | 213.23 ± 15.29 | 43.0 | 61.8% |
| LBP @ 0.40 | 96.82 | 195.27 ± 14.82 | 37.5 | 66.4% |
| LTP @ 0.50 | 96.82 | 185.46 ± 14.73 | 36.0 | 67.7% |

### Directional, n = 110

| arm | accuracy | steps ± SEM | median | converged <500 |
|---|---|---|---|---|
| control (HSV) | 89.09 | 275.83 ± 22.19 | 500.0 | 49.1% |
| LBP @ 0.50 | 93.64 | 260.65 ± 22.16 | 87.5 | 51.8% |
| LBP @ 0.40 | 94.55 | 245.38 ± 22.08 | 58.0 | 55.5% |
| LTP @ 0.50 | 94.55 | 243.96 ± 21.89 | 73.5 | 56.4% |

### Paired tests

| comparison | accuracy (McNemar) | steps (paired perm / Wilcoxon) | convergence (McNemar) |
|---|---|---|---|
| **ambient** ||||
| LTP vs control | 96.8 vs 93.2, 8–0, **p = 0.0078** | **p = 0.0001** / **p < 0.0001** | 67.7 vs 61.4, 16–2, **p = 0.0013** |
| LBP@0.50 vs control | 96.8 vs 93.2, 9–1, **p = 0.0215** | p = 0.538 / p = 0.212 | 61.8 vs 61.4, 9–8, p = 1.0 |
| LBP@0.40 vs control | 96.8 vs 93.2, 9–1, **p = 0.0215** | **p = 0.0065** / **p = 0.0021** | 66.4 vs 61.4, 15–4, **p = 0.0192** |
| LTP vs LBP@0.50 | 96.8 vs 96.8, 1–1, p = 1.0 | **p < 0.0001** / **p = 0.0010** | 67.7 vs 61.8, 13–0, **p = 0.0002** |
| **LTP vs LBP@0.40** | 96.8 vs 96.8, 1–1, p = 1.0 | p = 0.118 / p = 0.023 | 67.7 vs 66.4, 7–4, p = 0.549 |
| **directional** ||||
| LTP vs control | 94.5 vs 89.1, 6–0, **p = 0.031** | **p = 0.029** / **p = 0.044** | 56.4 vs 49.1, 11–3, p = 0.057 |
| LBP@0.50 vs control | 93.6 vs 89.1, 5–0, p = 0.063 | p = 0.181 / p = 0.341 | 51.8 vs 49.1, 5–2, p = 0.453 |
| LTP vs LBP@0.50 | 94.5 vs 93.6, 1–0, p = 1.0 | p = 0.200 / p = 0.110 | 56.4 vs 51.8, 9–4, p = 0.267 |
| **LTP vs LBP@0.40** | 94.5 vs 94.5, 1–1, p = 1.0 | p = 0.917 / p = 0.952 | 56.4 vs 55.5, 6–5, p = 1.0 |

Unpaired tests on steps are non-significant everywhere (e.g. ambient LTP vs control
p = 0.116 unpaired vs p = 0.0001 paired; paired r = 0.84). Pairing is what makes the
step effects visible, and the protocol section must say the rotations were shared.

## What this actually shows

1. **A local texture descriptor helps, and this is solid.** Both LBP and LTP beat the
   HSV-only control on accuracy in ambient (p ≈ 0.008–0.022, discordant 8–0 / 9–1),
   with the same direction under directional light. This is the paper's core claim and
   it survives.

2. **Fairly tuned, LBP is statistically indistinguishable from LTP.** At its own
   tolerance, LBP matches LTP's accuracy exactly in both conditions (96.82 / 94.55)
   and shows no reliable step difference: directional p = 0.92, ambient p = 0.118 by
   permutation (Wilcoxon p = 0.023 on the same data — mixed, so not something to lean
   on). Convergence rates match too (66.4% vs 67.7%, p = 0.55).

3. **The LTP-over-LBP advantage seen at tolerance 0.50 was a tuning artifact.** At the
   inherited 0.50, LTP looked clearly better on speed (ambient p < 0.0001, convergence
   discordant 13–0). Giving LBP its own threshold erased that. The residual LTP edge is
   ~10 steps ambient and ~1.4 steps directional.

**Consequence for the paper: LTP cannot be claimed to outperform LBP.** The defensible
claims are (a) texture beats color-only, significantly and in both lighting conditions,
and (b) LTP achieves this with the same tuning effort as LBP while being more robust to
the choice of tolerance — LTP's optimum sits on a flatter part of the curve, and LTP at
a single fixed 0.50 works well in both conditions whereas LBP needed re-tuning.

Suggested Discussion wording:

> We compared split LTP against classic LBP under identical sampling, encoding,
> weighting and matching, tuning each descriptor's tolerance on its own distance
> analysis. Both descriptors significantly improve recognition accuracy over the
> color-only baseline (ambient: 96.8% vs 93.2%, McNemar p = 0.022 for LBP and
> p = 0.008 for LTP, n = 220 paired episodes). With each tuned to its own optimum the
> two are statistically indistinguishable on both accuracy and steps to convergence,
> so we do not claim an advantage for the ternary formulation on these datasets. We
> note that LTP was less sensitive to the tolerance setting: a single value (0.50)
> was near-optimal under both lighting conditions, whereas LBP required re-tuning
> (0.50 → 0.40), and at the shared setting LTP converged faster (185.5 vs 213.2
> steps, paired p < 0.0001).

Do **not** reuse the earlier framing that the ternary split buys convergence speed —
that held only at the un-tuned LBP threshold.

## Files

- `src/tbp/monty/frameworks/utils/sensor_processing.py` — `lbp_codes`,
  `local_binary_pattern_and_hist`, `local_binary_pattern` dispatch key
- `src/tbp/monty/conf/monty/sensor_module/camera_surf_rgba_lbp.yaml`
- `src/tbp/monty/conf/monty/learning_module/evidence_1lm_nn5_dod0025_ltp_t040.yaml`
- `src/tbp/monty/conf/experiment/lbpcmp_{amb,dir}_train_11retex_lbp.yaml`
- `src/tbp/monty/conf/experiment/lbpcmp_{amb,dir}_eval_11retex_{noltp,ltp,lbp,lbp_t040}.yaml`
- `scripts/run_lbp_comparison_tuned.sh` — full study (train + eval + paired analysis)
