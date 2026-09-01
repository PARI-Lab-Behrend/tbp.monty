# LTP Texture Recognition: `tdcubs-may` vs `niels-refinements`

**Question:** Both branches implement a "Local Ternary/Binary Pattern" (LTP/LBP)
texture feature for Monty. Running the textured-object experiments
(`lbp_supervised_pre_training_texturedobs.yaml` for training,
`lbp_eval_texturedobs_dist_agent.yaml` for eval), recognition accuracy is **good
on `tdcubs-may` but poor on `niels-refinements`**. Why?

**Date:** 2026-06-13
**Branches compared:** `tdcubs-may` (current) vs `niels-refinements`
**Merge base:** `0acde3e2`

---

## TL;DR

The two branches are **not** running the same texture pipeline. They diverged from
different `main` bases and re-implemented the texture feature independently:

| | `tdcubs-may` | `niels-refinements` |
|---|---|---|
| Feature name | `local_binary_pattern` | `ltp` |
| Extractor | skimage `local_binary_pattern`, `method="uniform"` | custom split-LTP + ROR rotation-invariant encoding |
| Histogram size (8 neighbors) | **~10 bins** (uniform LBP) | **~72 bins** (pos + neg ROR codes) |
| Distance metric | **Hellinger**, bounded `[0, 1]` | **chi-square** (`cv2.HISTCMP_CHISQR`), unbounded `[0, ∞)` |
| Tolerance | **0.03** | **20.0** |
| Feature weight | **20** | **1** |

The accuracy gap is **not a bug** — it is a **tuning mismatch**. On `tdcubs-may` the
texture feature is configured to be the *dominant, highly discriminative* signal
(tight tolerance on a bounded metric, weight 20). On `niels-refinements` the
texture feature is configured so loosely (tolerance 20 on an unbounded metric,
weight 1) that it contributes **near-constant evidence to every stored node** and
therefore barely discriminates between objects.

Because the experiment's object set is dominated by **geometrically near-identical
objects that differ essentially only in surface texture** (baseball / tennis_ball
/ golf_ball; peach / plum / strawberry), texture *must* be the dominant
discriminator. `tdcubs-may` makes it so; `niels-refinements` does not — hence the
accuracy collapse.

> Note: the literal config files named above, and all of their LTP-specific
> building blocks, exist **only on `tdcubs-may`**. `niels-refinements` has its own
> equivalent configs (`supervised_pre_training_11retextured_obj_ltp.yaml`,
> `base_config_11retexturedobj_dist_agent_ltp.yaml`, etc.). The comparison below is
> therefore between *equivalent* pipelines, not byte-identical configs.

---

## 1. Why the feature evidence differs

Both branches compute per-node feature evidence the same way at the top level:

```
feature_evidence = clip(tolerance - distance, 0, inf) / tolerance      # in [0, 1]
node_score       = weighted_average(feature_evidence, weights)
```

The texture feature's contribution to `node_score` is governed by three knobs:
the **distance metric**, the **tolerance**, and the **weight**. All three differ.

### 1a. Distance metric — bounded vs unbounded

**`tdcubs-may`** — `DefaultFeatureEvidenceCalculator.hellinger_distance`
(`src/tbp/monty/frameworks/models/evidence_matching/feature_evidence/calculator.py`):

```python
return (1.0 / np.sqrt(2.0)) * np.linalg.norm(node_sqrt - query_sqrt, axis=1)
```

Hellinger distance on normalized histograms is **bounded in `[0, 1]`**: 0 = identical,
1 = disjoint. This makes the tolerance interpretable and stable.

**`niels-refinements`** — chi-square via OpenCV
(same file path, `HISTOGRAM_FEATURES = {"ltp"}` branch):

```python
chi_distances = np.array([
    cv2.compareHist(stored_hist.astype(np.float32),
                    query_hist.astype(np.float32),
                    cv2.HISTCMP_CHISQR)
    for stored_hist in stored_hists
])
```

`cv2.HISTCMP_CHISQR` computes `sum_i (H1_i - H2_i)^2 / H1_i`, which is:
- **unbounded** (`[0, ∞)`),
- **asymmetric** (denominator is the *stored* histogram only),
- **numerically sensitive** to sparse bins — small-mass bins in `H1` inflate terms.

### 1b. Tolerance — interacts with the metric's scale

- `tdcubs-may`: `local_binary_pattern` tolerance = **0.03** on a `[0, 1]` metric.
  Only textures with Hellinger distance < 0.03 produce *any* positive evidence —
  i.e. near-identical textures. Highly selective.
- `niels-refinements`: `ltp` tolerance = **20.0** on a `[0, ∞)` metric whose typical
  values for similar normalized histograms are small. Result:
  `clip(20 - chi², 0, ∞) / 20 ≈ 1.0` for **almost every node**, regardless of
  whether the texture matches. The texture term becomes an almost-constant ~1.0
  offset that carries little discriminative information.

(The `niels-refinements` displacement config even flags this as unverified:
`ltp: 1.0  # TODO check if this is reasonable`.)

### 1c. Feature weight — dominant vs negligible

In the weighted average over features:

- `tdcubs-may`: `local_binary_pattern` weight = **20**, vs `hsv ≈ [1, 0.5, 0.5]`,
  `pose_vectors = ones(3)`, `principal_curvatures_log = ones(2)`. Texture
  **dominates** the score.
- `niels-refinements`: `ltp` weight = **1**, comparable to hsv / curvature.
  Texture is one voice among many — and (per 1b) a near-constant one.

### Combined effect

| Branch | Texture term behavior | Net effect on recognition |
|---|---|---|
| `tdcubs-may` | sharp, near-binary "same texture?" gate, weighted ×20 | texture decides the match → distinguishes same-geometry objects |
| `niels-refinements` | ~1.0 for nearly all nodes, weighted ×1 | texture ≈ constant → same-geometry objects collapse together |

---

## 2. Why the histograms aren't directly comparable

The tolerance/weight values are **not portable** between branches because the two
histograms are different objects:

- **`tdcubs-may`** (`sensor_modules.py`, `_extract_and_add_features`):
  ```python
  ltp = local_binary_pattern(..., P=8, R=..., method="uniform", normalize=True)
  features["local_binary_pattern"] = ltp.histogram   # ~10 bins (P+2 for uniform)
  ```
  A coarse, 10-bin, rotation-invariant *uniform* LBP histogram.

- **`niels-refinements`** (`utils/sensor_processing.py`,
  `local_ternary_pattern_and_hist` / `ltp_codes` / `ror_encoding`):
  split-LTP (separate positive/negative codes, threshold = 5 on the 0–255 patch),
  each run through ROR rotation-invariant encoding, then concatenated:
  ```python
  histogram = concat(hist_pos, hist_neg)   # ~72 bins for n_neighbors=8
  histogram /= histogram.sum() + 1e-6
  ```
  A much finer ~72-bin histogram with most mass concentrated in a few bins (the
  threshold-5 "dead zone" makes many codes empty/sparse).

A many-bin, sparse histogram is exactly the regime where the asymmetric chi-square
distance is noisiest, and where a tolerance tuned for a 10-bin Hellinger histogram
is meaningless. Even with identical tolerance/weight numbers, the two pipelines
would behave differently; with the *different* numbers each branch actually uses,
they behave very differently.

---

## 3. Why this specific experiment exposes the difference

The eval object set (`eval_texturedobs_predefined.yaml`):

```
gelatin_box, potted_meat_can, sugar_box, foam_brick,
baseball, peach, tennis_ball, plum, golf_ball, strawberry
```

Several of these are **geometrically near-identical** — `baseball`, `tennis_ball`,
and `golf_ball` are all ~spheres; `peach`, `plum`, `strawberry` are all small
roundish objects. Morphological features (curvature, pose) **cannot** separate
them. The recent `tdcubs-may` work makes the intent explicit:

> `cde64528 Add YCB reskinned baseballs and softballs, config for eval on
> LBP-enabled model with no LBP-evidence contribution`

i.e. these are deliberately **reskinned** objects whose *only* distinguishing
signal is surface texture. Any pipeline where texture is non-dominant will fail to
tell them apart. That is precisely the `niels-refinements` configuration.

---

## 4. Recommendations to recover accuracy on `niels-refinements`

The fix is **retuning the `ltp` feature**, not changing the LTP extraction. In
priority order:

1. **Raise the `ltp` feature weight** in
   `conf/monty/learning_module/evidence_1lm_nn5_dod003_ltp.yaml` from `1` toward the
   range that makes texture dominant (start ~`20`, matching `tdcubs-may`'s LBP
   weight).
2. **Tighten the `ltp` tolerance** from `20.0` to a value where the texture term
   actually transitions between match/no-match across the object set. With
   chi-square this needs empirical calibration — measure the distribution of
   `cv2.compareHist(..., CHISQR)` between same-object and different-object patches
   and set tolerance near the separating value. (Expect something far below 20.)
3. **Consider swapping chi-square for the bounded Hellinger metric** used by
   `tdcubs-may`. A bounded `[0, 1]` distance makes the tolerance interpretable and
   removes the asymmetry/sparsity sensitivity. This is the lowest-risk way to make
   the `niels-refinements` texture term behave like the one that works.
4. **Update the displacement `graph_delta_thresholds` `ltp` value**
   (`displacement_delta_d0001_ltp.yaml`, currently `1.0  # TODO`) consistently with
   the chosen metric, so that learning stores texture-distinct points rather than
   collapsing them.

A quick first experiment: set `ltp` weight ≈ 20 and sweep tolerance over, e.g.,
`{0.5, 1, 2, 5}` for chi-square (or switch to Hellinger with tolerance ≈ 0.03–0.1),
and check accuracy on the reskinned-ball subset.

---

## 4a. Experiments created to test this (on `niels-refinements`)

A `HellingerFeatureEvidenceCalculator` was added
(`feature_evidence/calculator.py`) as a drop-in subclass of the default calculator
that compares histogram (`ltp`) features with the bounded Hellinger distance
instead of chi-square. It is selected per-config via the hypotheses updater's
`feature_evidence_calculator` argument — no other code changes, and the existing
chi-square default is untouched.

These eval experiments port the tdcubs-may texture-matching knobs (Hellinger
distance + tolerance + feature weight) and sweep tolerance/weight. **All reuse the
existing pretrained model** (`supervised_pre_training_11retextured_obj_ltp`) — these
are eval-time matching knobs, so no retraining is needed. Each has a distinct
`run_name` for separate analysis.

| Experiment config (`experiment=…`) | `run_name` | Hellinger tol | `ltp` weight |
|---|---|---|---|
| `base_config_11retexturedobj_dist_agent_ltp_hellinger_t003_w20` | `ltp_hellinger_t003_w20` | 0.03 | 20 |
| `base_config_11retexturedobj_dist_agent_ltp_hellinger_t010_w20` | `ltp_hellinger_t010_w20` | 0.10 | 20 |
| `base_config_11retexturedobj_dist_agent_ltp_hellinger_t020_w20` | `ltp_hellinger_t020_w20` | 0.20 | 20 |
| `base_config_11retexturedobj_dist_agent_ltp_hellinger_t010_w05` | `ltp_hellinger_t010_w05` | 0.10 | 5 |

Baseline for comparison: the existing `base_config_11retexturedobj_dist_agent_ltp`
(chi-square, tol 20, weight 1).

`t003_w20` is the literal tdcubs-may port; because the niels `ltp` histogram is finer
(~72 bins vs ~10), 0.03 may be too tight (everything → 0 evidence), so the
`t010`/`t020` rows loosen it and `t010_w05` checks weight sensitivity.

Run (one run name each):

```bash
conda activate tbp.monty
cd ~/Projects/tbp.monty
python run.py experiment=base_config_11retexturedobj_dist_agent_ltp_hellinger_t003_w20
python run.py experiment=base_config_11retexturedobj_dist_agent_ltp_hellinger_t010_w20
python run.py experiment=base_config_11retexturedobj_dist_agent_ltp_hellinger_t020_w20
python run.py experiment=base_config_11retexturedobj_dist_agent_ltp_hellinger_t010_w05
```

---

## 4b. Improved chi-square experiments (data-grounded)

Rather than keeping the loose default, the chi-square tolerance was calibrated
against the **actual stored histograms** in the pretrained 11-retextured model
(`supervised_pre_training_11retextured_obj_ltp`). The stored `ltp` feature occupies
72 bins (model `feature_mapping`: `ltp: [20, 92]`). The 11 objects are 4 reskinned
baseballs, 4 reskinned softballs, plus mug/bowl/banana — so within each ball group
the geometry is identical and **texture is the only discriminator**.

Measured chi-square (`cv2.HISTCMP_CHISQR`) distance distributions:

| Regime | mean | p25 | p50 | p75 | p90 | p95 |
|---|---|---|---|---|---|---|
| Within-object (same texture, different locations) | 0.66 | 0.08 | 0.20 | 0.48 | 0.97 | 1.53 |
| Across reskins (wrong texture, same shape — hardest) | 6.73 | 0.50 | 0.95 | 4.69 | 17.08 | 32.98 |
| Across ball groups | 4.42 | 0.43 | 0.91 | 2.16 | 5.68 | 20.12 |

Two takeaways:
1. The baseline tolerance of **20 is far too loose** — even the p90 of *wrong*-texture
   pairs (17) is below 20, so nearly every node scores ~max evidence. This is exactly
   why the baseline chi-square run can't discriminate.
2. A discriminative tolerance sits near the within-object **p75–p90 (≈0.5–1.0)**. At
   tol 0.5: a true match (median 0.20) → evidence `(0.5−0.2)/0.5 = 0.6`, while a wrong
   reskin (median 0.95) → 0. (Note the within/across distributions overlap somewhat,
   so chi-square can't separate them perfectly — hence also trying Hellinger.)

Experiments (all weight 20, chi-square = default calculator, eval-only, same
pretrained model, distinct run names):

| Experiment config (`experiment=…`) | `run_name` | chi-sq tol | `ltp` weight |
|---|---|---|---|
| `base_config_11retexturedobj_dist_agent_ltp_chisq_t050_w20` | `ltp_chisq_t050_w20` | 0.50 (recommended) | 20 |
| `base_config_11retexturedobj_dist_agent_ltp_chisq_t100_w20` | `ltp_chisq_t100_w20` | 1.00 | 20 |
| `base_config_11retexturedobj_dist_agent_ltp_chisq_t200_w20` | `ltp_chisq_t200_w20` | 2.00 | 20 |

```bash
python run.py experiment=base_config_11retexturedobj_dist_agent_ltp_chisq_t050_w20
python run.py experiment=base_config_11retexturedobj_dist_agent_ltp_chisq_t100_w20
python run.py experiment=base_config_11retexturedobj_dist_agent_ltp_chisq_t200_w20
```

---

## 5. Caveats

- This analysis is from **code and config inspection only**; no experiments were
  re-run. The directional claims (texture is dominant on `tdcubs-may`, near-constant
  on `niels-refinements`) follow directly from the metric/tolerance/weight settings;
  the *exact* chi-square magnitudes should be confirmed empirically before finalizing
  new tolerance values (recommendation 2).
- The branches differ in far more than the texture feature (`niels-refinements` is
  built on a newer `main` with the percept/Salience/MuJoCo refactors). Those changes
  are unlikely to explain a *texture-discrimination* accuracy gap, but if retuning
  the `ltp` feature does not close the gap, the next place to look is the
  feature-change / graph-delta filtering that decides which points get stored during
  learning.

---

## Appendix — key source locations

**`tdcubs-may`**
- Feature evidence (Hellinger): `src/tbp/monty/frameworks/models/evidence_matching/feature_evidence/calculator.py`
- LBP extraction (skimage uniform, 10 bins): `src/tbp/monty/frameworks/models/sensor_modules.py` (`local_binary_pattern` branch in `_extract_and_add_features`)
- SM config: `conf/monty/sensor_module/camera_lbp_dist.yaml`
- LM config (weight 20, tol 0.03): `conf/monty/learning_module/evidence_lbp_1lm_nn5_dod003.yaml`
- Displacement config: `conf/monty/learning_module/lbp_displacement_delta_d0001.yaml`
- Experiments: `conf/experiment/lbp_supervised_pre_training_texturedobs.yaml`, `conf/experiment/lbp_eval_texturedobs_dist_agent.yaml`
- Objects: `conf/env_interface/{train,eval}_texturedobs_predefined.yaml`

**`niels-refinements`**
- Feature evidence (chi-square): `src/tbp/monty/frameworks/models/evidence_matching/feature_evidence/calculator.py` (`HISTOGRAM_FEATURES = {"ltp"}`)
- LTP extraction (split-LTP + ROR, ~72 bins): `src/tbp/monty/frameworks/utils/sensor_processing.py` (`get_ltp_texture_feature_vector`, `local_ternary_pattern_and_hist`, `ltp_codes`, `ror_encoding`)
- SM config: `conf/monty/sensor_module/camera_dist_ltp.yaml`
- LM config (weight 1, tol 20): `conf/monty/learning_module/evidence_1lm_nn5_dod003_ltp.yaml`
- Displacement config (`ltp: 1.0  # TODO`): `conf/monty/learning_module/displacement_delta_d0001_ltp.yaml`
- Experiments: `conf/experiment/supervised_pre_training_11retextured_obj_ltp.yaml`, `conf/experiment/base_config_11retexturedobj_dist_agent_ltp.yaml`
