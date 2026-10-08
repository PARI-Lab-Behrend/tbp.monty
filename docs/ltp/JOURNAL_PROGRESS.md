# Progress log: journal experiment plan

Tracks completion of [JOURNAL_EXPERIMENT_PLAN.md](JOURNAL_EXPERIMENT_PLAN.md).
Update the status table and add a dated log entry whenever something finishes,
fails, or a decision is made. Numbers go here only once regenerated under the
plan (prior BICA-era results are ignored).

Status legend: **done** · **in progress** · **todo** · **blocked** · **n/a**

Last updated: 2026-10-07

## At a glance

In execution order. "Waits on" names what must finish before a step can run.

| Step | Status | Waits on | Gate / output | Note |
|---|---|---|---|---|
| Build tooling (`material_stats.py`, `make_recipes.py`) | done | -- | `material_stats.csv`, recipe YAMLs | 3DC-Texture design approved by user 2026-10-04 |
| Phase 2 -- scale / footprint sweep (ambient) | done | -- | `target_size` decision | **0.12 m kept** (user decision); rule would pick 0.25 m by a small margin |
| Datasets 3DC-Texture (50) + 3DC-Mixed (60) | done, frozen | -- | `$MONTY_DATA/compat3d_texture`, `compat3d_mixed` | Shapes chosen once by `select_dataset_shapes.py`; recipes frozen |
| Prelim A -- asset sanity | done | -- | 100% pass + contact sheet | Passed via selection; final static checks clean |
| Gabor feature (`gabor`, `gabor_rgb`, `gabor_hsv`) | done | -- | functions + tests + FFT path | 21 new tests; full unit suite passes |
| Prelim C -- smoke test + compute budget | done | -- | budget in this doc | All gates pass; combined-feature training confirmed equivalent |
| Main trainings (4 combined-feature: 2 datasets x 2 lightings) | **done** (ambient 2026-10-05 22:19, directional 2026-10-06 09:52) | -- | `pretrain_3dc_{texture,mixed}_{amb,dir}_all_features` models | Ambient ~22:30; then Phase 4 (amb) + Prelim B + Phase 3 evals (~2 h), then directional (~9.5 h). Driver `journal_logs/run_after_ambient.sh`, status `main_trainings.status` |
| Prelim B -- headroom | done (3DC-Texture `hsv` 85% > 80%: dataset kept, user 2026-10-07) | main trainings (ambient) | `hsv` baseline 30-80% on 3DC-Texture | `prelim_eval_3dc_<dataset>_amb_{shape_only,hsv}_2rot` |
| Phase 3 -- color space (`ltp_rgb` vs `ltp_hsv`, ambient) | **done: `ltp_rgb`** | main trainings (ambient), Phase 4 (amb) | decision record | `prelim_eval_3dc_<dataset>_amb_{ltp,ltp_rgb,ltp_hsv}_2rot`, at ambient-calibrated tolerances |
| Phase 4 -- tolerance calibration | **done (both lightings)** | main trainings | tolerance per feature, per dataset x lighting | `scripts/phase4_calibration.py` -> `docs/ltp/results/phase4/tolerances.csv`; generator writes them into the eval configs. Regime pending user (`across_object` used for now) |
| **Phase 5 -- main comparison: ambient, then directional** | **todo -- next (new chat)** | -- | accuracy per scenario (both sets), steps on 3DC-Mixed; McNemar/Wilcoxon, interaction CI | ~7 h per lighting at 5 rotations x 2 seeds, ~20 processes |
| Phase 6 -- wall time | todo | Phase 5 (patch dumps) | per-patch µs, per-step, time-to-correct | Per-step harness built (`scripts/phase6_step_timing.py`); micro (per-patch) benchmark not yet |
| Phase 7 -- robustness (ambient) | todo | Phase 5 | noise / footprint mismatch / threshold | 0.06 m mismatch set to build |
| Stats, figures, writing | todo | Phase 5-7 | | |

## Next session: start Phase 5

Everything Phase 5 needs exists; nothing is running.

- Models: `~/tbp/results/monty/pretrained_models/my_trained_models/pretrain_3dc_{texture,mixed}_{amb,dir}_all_features/`.
- Eval configs: `experiment/journal/eval_3dc_<dataset>_<lighting>_<condition>`,
  carrying each model's Phase 4 tolerance and `save_eval_models: false`.
  Phase 5's 8 conditions: `shape_only`, `hsv`, `ltp`, `ltp_rgb`,
  `hsv_plus_ltp_rgb` (proposed), `gabor`, `gabor_rgb`, `hsv_plus_gabor_rgb`.
- Run with `run_parallel.py`, **<= ~20 processes in total** (memory), and
  `OMP_NUM_THREADS=1` (and MKL / OpenBLAS) or load explodes. Second seed:
  `experiment.config.seed=43`, run name suffix `_seed43` (what
  `scripts/phase5_stats.py` pools). Pattern and memory watchdog:
  `~/tbp/results/monty/journal_logs/run_prelim_evals_then_dir_v2.sh`.
- `run_parallel.py` aborts a whole run if one worker raises, and hangs if a
  worker is killed (e.g. out of memory); check every run for a top-level
  `eval_stats.csv`.
- Still to do for the directional scenario: evaluate the directional models
  at the *ambient* tolerances too (plan, Phase 5), i.e. extra configs or
  overrides.
- Expect little steps headroom on 3DC-Mixed: Phase 3 / Prelim B medians are
  26-28.5 steps in every condition (shape only 27).
- Analysis: `scripts/phase5_stats.py --root ~/tbp/results/monty/projects/journal --dataset {texture,mixed} --color-ltp ltp_rgb --out docs/ltp/results/phase5`.

## Inventory (updated 2026-10-05)

| Item | State | Where |
|---|---|---|
| Feature keys `ltp`, `ltp_rgb`, `ltp_hsv` | present | `frameworks/utils/sensor_processing.py`, `frameworks/models/sensor_modules.py` |
| Offline distance analysis (`tolerance_sweep`, `best_tolerance`, Hellinger + 6 others) | present | `frameworks/utils/ltp_distance_analysis.py` |
| Location-specificity AUC function (Phase 2 measure 2) | present (+ unit tests) | `ltp_distance_analysis.location_specificity_auc`; analysis script `scripts/phase2_footprint_analysis.py` |
| `compat3d_helper.py --recipe ... --build` | present | `~/3dcompat-habitat-prep/` |
| Source zip `3DCoMPaT_ZIP.zip` | present | `~/3dcompat-habitat-prep/3DCoMPaT_3D/` |
| Old 11-object sets `compat3d_ltp`, `compat3d_habitat_test` | present | `~/tbp/data/` (= `$MONTY_DATA`, exported in `~/.bashrc`) |
| Env / train / eval configs for `compat3d_ltp` | present, uncommitted | `conf/env_interface/*3dcompat*`, `conf/environment/habitat_3dcompat_*` |
| Sensor-patch renderer (distant agent at 0.3 m -- *not* the surface-agent experiments' view) | present | `scripts/compat3d_test/render_sensor_snapshots.py` |
| Trusted sensor-view capture (`show_sensor_output` frames, headless) | present | `scripts/compat3d_test/capture_sensor_output.py` |
| Material stats + recipe generator | present (committed on branch `journal-datasets`) | `~/3dcompat-habitat-prep/{material_stats.py,make_recipes.py,material_stats.csv,recipes/}` |
| Configs for 3DC-Texture / 3DC-Mixed (12 conditions x 2 lightings, 4 pretrainings) | present | generated by `scripts/generate_journal_configs.py`; `experiment/journal/{pretrain,eval}_3dc_*` |
| Ambient light setup `AMBIENT_LIGHTS`; directional = Habitat default (`lights: None`) | present | `simulators/habitat/simulator.py` |
| Ambient and directional 3DCoMPaT environments (explicit `lights` in both) | present | `environment/habitat_3dcompat_journal_surf_agent_{amb,dir}.yaml`. The older `compat3d_ltp` configs set no `lights` key, so they run directional |
| Gabor feature | present | `sensor_processing.py`, `sensor_modules.py`, evidence calculator |
| Per-step timing harness (Phase 6) | present | `scripts/phase6_step_timing.py` (serial `run.py`; one JSON line per episode) |
| Phase 4 calibration (Monty's distances: Hellinger for LTP, Euclidean / sqrt(channels) for Gabor; gap-optimal tolerance, `across_object` and `same_shape` regimes) | present | `scripts/phase4_calibration.py`; reproduces Phase 2's 0.12 m `ltp_rgb` result exactly (same-shape tolerance 0.26, gap 0.312, d' 1.15) |
| Phase 5 statistics (summary, McNemar / Wilcoxon + Holm, bootstrap CIs, interaction, steps-to-correct curves) | present (checked on synthetic runs) | `scripts/phase5_stats.py`; pairs episodes by (object, `episode_seed`); seed runs named `eval_3dc_<dataset>_<lighting>_<condition>_seed<N>` are pooled |
| Prelim A / selection / Prelim C tooling | present | `scripts/compat3d_test/{prelim_a_walk,prelim_a_checks,select_dataset_shapes}.py`, `scripts/prelim_c_eval.py` |

### Journal configs (old 11-object set, `compat3d_ltp`)

All journal experiments live in `conf/experiment/journal/` (run as
`experiment=journal/<name>`); learning modules used only by them live in
`conf/monty/learning_module/journal/`. Snapshots: `tests/conf/snapshots/journal/`.
`3dcompat11` = the 11-object `compat3d_ltp` set; the new sets will get their own
names. Every texture feature has weight 1.0; tolerances are the BICA values until
Phase 4. Columns read from the configs' `defaults` and resolved snapshots.

| Center-pixel `hsv` | Texture feature (weight, tolerance) | Eval config | Pretrained model |
|---|---|---|---|
| off | none | `randrot_3dcompat11_shape_only` | `surf_pre_training_3dcompat11_ltp` |
| on | none | `randrot_3dcompat11_hsv` (baseline) | `surf_pre_training_3dcompat11_ltp` |
| off | `ltp` (1.0, 0.50) | `randrot_3dcompat11_ltp` | `surf_pre_training_3dcompat11_ltp` |
| off | `ltp_rgb` (1.0, 0.23) | `randrot_3dcompat11_ltp_rgb` | `surf_pre_training_3dcompat11_ltp_rgb` |
| on | `ltp_rgb` (1.0, 0.23) | `randrot_3dcompat11_hsv_plus_ltp_rgb` | `surf_pre_training_3dcompat11_ltp_rgb` |
| off | `ltp_hsv`, circular hue (1.0, 0.23) | `randrot_3dcompat11_ltp_hsv` | `surf_pre_training_3dcompat11_ltp_hsv` |
| off | `ltp_hsv`, linear hue (1.0, 0.23) | `randrot_3dcompat11_ltp_hsv_linear_hue` | `surf_pre_training_3dcompat11_ltp_hsv_linear_hue` |

## Decisions

Record each with date, the evidence, and the rule applied.

| Decision | Value | Date | Evidence |
|---|---|---|---|
| `target_size` (patch footprint) | **0.12 m** (the existing default) | 2026-10-04 | User decision, overriding the Phase 2 rule (which picks 0.25 m: evidence gap 0.353 vs 0.312, d' 1.31 vs 1.15). The margin is consistent over sampling seeds but untested over retraining (1 training per size), and 0.12 passes the location floor (AUC 0.71). Revisit if texture separation causes problems. Phase 7 mismatch sets: 0.06 m (0.5x, to build) and 0.25 m (2x, built) |
| 3DC-Texture design | dataset composition + 4 same-family swaps of the most visible slot (>= 40% of visible surface), closest in mean color, texture energy >= 5 | 2026-10-04 | User rejected one-material-per-object (unrealistic); approved this after Habitat previews and `show_sensor_output` frames |
| Patch footprint at `target_size` 0.12 | 3.9 mm of surface (~3% of object) | 2026-10-04 | Raw sensor arrays from a real ottoman episode; judged usable by user. Phase 2 still decides `target_size` |
| Color LTP winner (`ltp_rgb` or `ltp_hsv`) | **`ltp_rgb`** | 2026-10-06 | Phase 3 rule: `ltp_hsv` only if >= 3 points better on both sets; it was +0.0 (Texture) / +0.8 (Mixed) |
| Tolerances per feature (ambient) | see Phase 4 results | 2026-10-05 | Gap rule against all other objects, per model (kept 2026-10-07, plan's rule); same-shape column reported for 3DC-Texture |
| Dataset shape selection rule | 200-step ambient walks from [0,0,0] and [90,0,0]: on-object > 0.9 (mean); 3DC-Texture also slot visits >= 0.4 and every variant's slot brightness >= 0.15 | 2026-10-04 | User-approved; run once, recipes frozen. Log: `selection_log.csv` (prep `recipes/`, copy in `docs/ltp/results/prelim_a/`) |
| 3DC-Texture classes | table, love_seat, stool (wood); sofa, bed (fabric); chair, ottoman (leather); vase, jug (metal); toilet (marble) | 2026-10-04 | dresser and cabinet failed all 30 candidates each; user chose to swap in classes; love_seat and stool were the first two passing (garbage_bin, bag also passed, not needed) |
| 3DC-Mixed classes | 28 classes, 60 shapes | 2026-10-04 | shelf and skateboard had no passing shape in 20; cabinet, dishwasher, dresser 1 each; 7 classes topped up to 3 to keep 60 |
| Primary lighting scenario | ambient (`AMBIENT_LIGHTS`); directional is the second scenario | 2026-10-04 | User decision: ambient is the default all phases run in |
| Tolerances per feature, directional | see Phase 4 directional results | 2026-10-06 | Same rule on the directional models; Phase 5 also evaluates them at the ambient tolerances |
| Directional-scenario conditions | all 8 Phase 5 conditions (provisional) | | Cut to baseline / proposed / `ltp` / Gabor matched if the Prelim C budget requires |
| Trainings | 1 combined-feature run per dataset x lighting (4 total) | 2026-10-04 | Texture features don't enter `graph_delta_thresholds`; gate: Prelim C equivalence check |
| Rotations x seeds for Phase 5 | **5 x 2** | 2026-10-07 | Kept as planned. Memory (not cores) caps evals at ~20 processes (~3-5 GB each): ~7 h wall per lighting, ~14 h for both |
| 3DC-Texture headroom (`hsv` 85% > 80% ceiling) | keep the dataset; report the deviation | 2026-10-07 | User decision; miss is inside the 95% CI (~77-91%) and steps/timeouts leave room |
| Where the speed claim is tested | 3DC-Mixed only | 2026-10-07 | User decision: on 3DC-Texture 94-98% of episodes hit the 500-step cap in every condition; 3DC-Texture reports accuracy. Stopping rule and weights unchanged |

## Phase 3 results (2026-10-06, ambient, 2 rotations x seed 42)

Center-pixel `hsv` off in every row, so the LTP variant is the only non-shape
feature; tolerances from Phase 4 (ambient, all-other-objects column), weight
1.0. Same episodes (object, rotation) in every row. Runs
`prelim_eval_3dc_<dataset>_amb_<condition>_2rot`; CSVs and confusion matrices
in `docs/ltp/results/phase3/`.

| Dataset | Texture feature | Correct | Converged before cap | Timed out | Median steps | Mean steps |
|---|---|---|---|---|---|---|
| 3DC-Texture | none (shape only) | 21% | 1% | 98% | 500 | 490 |
| 3DC-Texture | `ltp` (grayscale) | 67% | 2% | 96% | 500 | 485 |
| 3DC-Texture | `ltp_rgb` | **97%** | 5% | 94% | 500 | 477 |
| 3DC-Texture | `ltp_hsv` | 97% | 4% | 95% | 500 | 482 |
| 3DC-Mixed | none (shape only) | 94.2% | 94.2% | 0% | 27 | 30.8 |
| 3DC-Mixed | `ltp` | 95.8% | 95.8% | 0% | 28.5 | 31.5 |
| 3DC-Mixed | `ltp_rgb` | 99.2% | 97.5% | 1.7% | 27 | 29.6 |
| 3DC-Mixed | `ltp_hsv` | 100% | 100% | 0% | 26.5 | 28.6 |

Paired `ltp_hsv` vs `ltp_rgb` (same object x rotation): 3DC-Texture 0.0
points (2 vs 2 discordant, exact McNemar p = 1; steps +4.5, Wilcoxon p =
0.46); 3DC-Mixed +0.8 points (1 vs 0, p = 1; steps -1.0, p = 0.18).

- **Decision (pre-registered rule): `ltp_rgb`.** `ltp_hsv` needed >= 3
  points on both sets; it has 0.0 and 0.8. `ltp_hsv` stays as an ablation row.
- Color matters on 3DC-Texture: `ltp_rgb` 97% vs grayscale `ltp` 67% (and
  the center-pixel `hsv` baseline 85%, Prelim B).
- **Steps problem on 3DC-Texture:** every condition, even at 97% correct,
  reaches the 500-step cap in 94-98% of episodes, i.e. Monty almost never
  becomes confident enough to stop between the 5 same-shape variants; those
  episodes are scored by the most likely hypothesis at the cap. So on this
  set "fewer steps" cannot show up in median steps at the current settings;
  only accuracy-at-cap (or evidence-margin / steps-to-correct-MLH measures)
  can. Likely cause: Monty stops when the top hypothesis leads the next by
  `x_percent_threshold` (20%) of its evidence, and the variants differ only
  in one texture feature of weight 1.0 next to pose and curvature terms, so
  the per-step evidence gap is small (Phase 4 gaps 0.15-0.22) against a total
  that grows every step. Code check (2026-10-06): the rule is confirmed --
  an object is the unique match only when every other object's evidence is
  below the leader's minus `x_percent_threshold` (20%) of it
  (`EvidenceGraphLM`, docstring at `learning_module.py:171`). Because evidence
  is summed over steps, the leader's relative margin tends to (per-step
  evidence gap) / (per-step evidence); same-shape variants get equal pose and
  curvature evidence, so if the texture term's share keeps that ratio under
  20%, episodes cannot terminate at *any* step count, not merely within 500.
  The size of that ratio is not yet measured. Options (user decision, before
  Phase 5): (a) keep settings; report accuracy at the cap plus "steps until
  the target first becomes and stays the most likely hypothesis" from the
  per-step logs as the speed measure; (b) change the stopping rule or the
  texture weight for every condition (a protocol change, applied equally).

## Prelim B results (2026-10-06, ambient, 2 rotations x seed 42)

Runs `prelim_eval_3dc_<dataset>_amb_{shape_only,hsv}_2rot`, every object x 2
random rotations, 500 matching steps max, Phase 4 tolerances not involved (no
texture feature). "Correct" counts `correct` + `correct_mlh` (most likely
object at the step cap), as Monty's `percent_correct`. "Right shape/class" =
the most likely object has the target's shape (3DC-Texture) or class
(3DC-Mixed, where every object is a distinct shape). Script
`scripts/prelim_b_headroom.py`; CSVs and confusion matrices in
`docs/ltp/results/prelim_b/`.

| Dataset | Center-pixel `hsv` | Episodes | Correct | Converged before cap | Right shape/class | Timed out | Median steps |
|---|---|---|---|---|---|---|---|
| 3DC-Texture | off (shape only) | 100 | 21% | 1% | 100% | 98% | 500 |
| 3DC-Texture | on (baseline) | 100 | **85%** | 2% | 100% | 96% | 500 |
| 3DC-Mixed | off (shape only) | 120 | 94.2% | 94.2% | 95.8% | 0% | 27 |
| 3DC-Mixed | on (baseline) | 120 | 99.2% | 98.3% | 99.2% | 0.8% | 26 |

3DC-Texture `hsv` accuracy per shape: bed, sofa, toilet 100%; jug,
love_seat, ottoman, stool 90%; vase 80%; chair 70%; table 40%. Shape only:
20% on every shape (chance among 5 variants; 30% vase).

Against the pre-registered criteria:
- 3DC-Texture shape only: **pass** -- chance among the variants, shape always
  right.
- 3DC-Texture `hsv` in [30, 80]%: **fails narrowly** -- 85% (100 episodes;
  binomial 95% CI ~77-91%). The color-separable variants are bed, sofa and
  toilet (the plan's risk list named the sofa). But `hsv` almost never
  *converges*: 96% of episodes reach the 500-step cap and are scored by the
  most likely hypothesis, so the headroom for "fewer steps" is near-total.
- 3DC-Mixed shape only > 70%: **pass** (94%).
- 3DC-Mixed `hsv` not at 100% with no steps to spare: **marginal** -- 99.2%
  at a median 26 steps (shape only 27): little room on accuracy or steps;
  this set can show "does not hurt", as the plan intends, not "helps".

**Decision needed (user):** the plan's remedy for `hsv` > 80% is to replace
the variants `hsv` separates (bed, sofa, toilet) and rebuild 3DC-Texture,
which means retraining its ambient and directional models (~9 h each).

## Phase 4 results, directional (2026-10-06)

Same procedure and seed as the ambient table, on the directional models
(Habitat's default lights: strong, pose-dependent shading). The ambient rows
reproduced exactly when all four models were recalibrated together. The eval
configs now carry each model's own tolerance (40 evals).

| Dataset | Feature | Tolerance (all other objects) | Gap | d' | Tolerance (same shape) | Gap | d' | Gap vs ambient |
|---|---|---|---|---|---|---|---|---|
| 3DC-Texture | `ltp` | 0.349 | 0.101 | 0.33 | 0.368 | 0.072 | 0.23 | -50% |
| 3DC-Texture | `ltp_rgb` | 0.329 | 0.122 | 0.42 | 0.355 | 0.077 | 0.25 | -44% |
| 3DC-Texture | `ltp_hsv` | 0.362 | 0.180 | 0.72 | 0.315 | 0.100 | 0.38 | -36% |
| 3DC-Texture | `gabor` | 0.750 | 0.087 | 0.40 | 0.697 | 0.041 | 0.18 | -15% |
| 3DC-Texture | `gabor_rgb` | 0.766 | 0.087 | 0.41 | 0.694 | 0.040 | 0.18 | -15% |
| 3DC-Texture | `gabor_hsv` | 0.706 | 0.093 | 0.43 | 0.572 | 0.046 | 0.23 | -14% |
| 3DC-Mixed | `ltp` | 0.385 | 0.155 | 0.46 | -- | | | -43% |
| 3DC-Mixed | `ltp_rgb` | 0.379 | 0.162 | 0.49 | -- | | | -42% |
| 3DC-Mixed | `ltp_hsv` | 0.376 | 0.216 | 0.84 | -- | | | -33% |
| 3DC-Mixed | `gabor` | 0.612 | 0.078 | 0.34 | -- | | | -18% |
| 3DC-Mixed | `gabor_rgb` | 0.615 | 0.078 | 0.34 | -- | | | -18% |
| 3DC-Mixed | `gabor_hsv` | 0.567 | 0.082 | 0.37 | -- | | | -20% |

("Gap vs ambient" uses the all-other-objects gap.)

- Directional light costs LTP 33-50% of its offline separation and Gabor
  14-20%. `ltp_rgb` still leads `gabor_rgb` (0.122 vs 0.087 on 3DC-Texture,
  0.162 vs 0.078 on 3DC-Mixed), but by much less than under ambient light.
  Plausible reason: LTP compares each pixel with its neighbors against a
  fixed intensity threshold (5), so shading gradients across the patch flip
  codes; Gabor responses are normalized per channel. Phase 5 tests whether
  this offline drop shows up online.
- Tolerances rise 10-25% under directional light (features vary more within
  an object).

## Phase 4 results, ambient (2026-10-05)

Tolerance for each texture feature, chosen on each ambient model's own
training graphs by the gap rule: the tolerance that maximizes mean feature
evidence for nodes of the same object minus mean evidence for nodes of other
objects. "Gap" is that difference at the chosen tolerance (0 to 1; larger =
the feature separates objects better per step); d' is the gap divided by the
pooled spread of the evidence. Distances are the ones Monty matches with:
Hellinger for LTP histograms, Euclidean / sqrt(channels) for Gabor vectors.
`same_shape` compares against the other 4 variants of the same shape only,
so it exists only for 3DC-Texture. 1,000 nodes per object, 2,000 query nodes,
seed 0. CSV: `docs/ltp/results/phase4/tolerances.csv`.

| Dataset | Feature | Tolerance (all other objects) | Gap | d' | Tolerance (same shape) | Gap | d' |
|---|---|---|---|---|---|---|---|
| 3DC-Texture | `ltp` | 0.305 | 0.200 | 0.69 | 0.287 | 0.145 | 0.48 |
| 3DC-Texture | `ltp_rgb` | 0.297 | 0.219 | 0.79 | 0.284 | 0.153 | 0.52 |
| 3DC-Texture | `ltp_hsv` | 0.306 | 0.282 | 1.07 | 0.250 | 0.186 | 0.67 |
| 3DC-Texture | `gabor` | 0.728 | 0.102 | 0.45 | 0.664 | 0.047 | 0.20 |
| 3DC-Texture | `gabor_rgb` | 0.737 | 0.102 | 0.46 | 0.654 | 0.046 | 0.20 |
| 3DC-Texture | `gabor_hsv` | 0.684 | 0.108 | 0.48 | 0.543 | 0.055 | 0.27 |
| 3DC-Mixed | `ltp` | 0.351 | 0.272 | 0.84 | -- | | |
| 3DC-Mixed | `ltp_rgb` | 0.354 | 0.280 | 0.88 | -- | | |
| 3DC-Mixed | `ltp_hsv` | 0.332 | 0.322 | 1.17 | -- | | |
| 3DC-Mixed | `gabor` | 0.594 | 0.095 | 0.39 | -- | | |
| 3DC-Mixed | `gabor_rgb` | 0.583 | 0.095 | 0.41 | -- | | |
| 3DC-Mixed | `gabor_hsv` | 0.522 | 0.103 | 0.45 | -- | | |

- The eval configs use the "all other objects" column (the plan's rule).
  The same-shape choice would only change 3DC-Texture, by <= 0.06 for LTP
  and 0.06-0.14 for Gabor.
- Offline, every LTP variant separates objects about twice as well per step
  as its Gabor counterpart (gap 0.20-0.32 vs 0.10). `ltp_hsv` has the
  largest gap on both sets -- an offline hint for Phase 3, which decides
  online.
- Every tolerance is above the provisional values (LTP 0.23-0.50 -> ~0.3;
  Gabor placeholder 0.30 -> 0.52-0.74).

## Phase 2 results (2026-10-04, ambient, `ltp_rgb`)

15 objects (jug / metal, ottoman / leather, table / wood x 5 variants), 2
rotations each, one pretraining per size. Separability = Monty's Hellinger
distance between nodes of the same object vs nodes of another variant of the
*same shape*, scored by the evidence gap at the gap-optimal tolerance.
Location AUC = probability that node pairs > 5 cm apart differ more in
`ltp_rgb` than pairs < 1 cm apart (0.5 = no location information). Values are
sampling seed 0; evidence gap for seeds 1 and 2 in parentheses. CSVs:
`docs/ltp/results/phase2/`.

| `target_size` | Patch vs object | Evidence gap | d' | Best tolerance | Location AUC, median (min) | Location AUC: jug / ottoman / table |
|---|---|---|---|---|---|---|
| 0.12 m | 3.9 mm patch (measured), ~3% of object | 0.312 (0.315, 0.306) | 1.15 | 0.26 | 0.71 (0.59) | 0.74 / 0.73 / 0.61 |
| **0.25 m** | same patch, ~1.6% of object | **0.353** (0.367, 0.354) | **1.31** | 0.30 | 0.73 (0.59) | 0.77 / 0.74 / 0.62 |
| 0.5 m | same patch, ~0.8% of object | 0.327 (0.331, 0.323) | 1.13 | 0.28 | 0.77 (0.56) | 0.82 / 0.81 / 0.71 |

- All three sizes pass the location-specificity floor (median AUC > 0.6), so
  the rule reduces to the largest separability: 0.25 m. **Not adopted**: the
  user judged the margin not significant and kept 0.12 m (see Decisions).
- Location specificity rises with size (more distinct surface per absolute
  cm); separability peaks in the middle.
- Caveats: one training per size (training-to-training variance unmeasured);
  3 shapes; the table is the weakest object for location AUC at every size.
- Compute: `run_parallel.py` (one process per object) took ~11 min per size,
  all three concurrently; the sequential `run.py` pace was ~40 min per size.

## Prelim A results (2026-10-04, 0.12 m)

**Final (frozen datasets):** static checks clean (no duplicates, no
`placeholder`, sizes 0.12 m +- 2%). On-object from start rotation [0,0,0]
alone: median 1.00 (Texture) / 0.99 (Mixed); 1 Texture shape (table, 0.85) and
8 Mixed objects (0.81-0.90) are below 0.9 from that single start but passed the
2-start mean. Patch brightness median V: ambient 0.27 / 0.36, directional
0.24 / 0.29; objects with any nearly-black step: ambient 5 / 5, directional
14 / 17. Selection tried 102 Texture candidates and 295 Mixed candidates (log rows).
CSVs: `docs/ltp/results/prelim_a/*_final.csv`, `selection_log.csv`.

**Before selection** (first recipes; kept for the record, `*_preselection.csv`):

`scripts/compat3d_test/prelim_a_checks.py`: static checks plus a real 200-step
surface-agent walk per object (rotation [0,0,0]) under each lighting. CSVs in
`docs/ltp/results/prelim_a/`.

- **Static: all pass** -- no duplicates, no `placeholder`, every max extent
  0.12 m +- 2%, every object loads and walks.
- **On-object (patch centre on the object, pass > 0.9): fails widely.**
  3DC-Texture median 0.79, 30/50 objects fail; 3DC-Mixed median 0.86, 31/60
  fail. The metric is sound: on the old 11-object set the solid ottoman,
  bench and table score 0.99-1.0, its open-frame chair 0.57 and boat 0.32.
- **3DC-Texture per shape** (on-object; do the 5 variants' patch statistics
  differ under ambient light?): dresser 0.81, no; table 0.04, no (patch stays
  on the crossed legs); cabinet 0.99, no; sofa 0.77, yes; bed 0.58, no;
  chair 1.00, yes; ottoman 0.59, yes (seat reached only late); vase 0.58, yes;
  jug 1.00, yes; toilet 1.00, yes. "No" = the walk essentially never reached
  the varied slot.
- **Brightness:** median patch V, ambient 0.28 (Texture) / 0.34 (Mixed);
  directional 0.19 / 0.30. Under directional light the chair leathers drop to
  V 0.02-0.06 (black); nearly-black steps appear for 18/50 Texture objects
  (dir) and 13-17/60 Mixed objects.
- Caveat: one start rotation, 200 steps; training uses 14 rotations x 1000
  steps, eval random rotations x up to 500 steps.

## Prelim C results (2026-10-05, ambient, 3DC-Texture)

Tiny runs: 3 objects (two ottoman variants + a jug), 1 training rotation
([0,0,0]), 1 random eval rotation, seed 42. Configs from
`scripts/generate_journal_configs.py`; per-feature evidence recorded with
`scripts/prelim_c_eval.py`. Accuracy on 3 objects means nothing and is not
reported.

| Gate | Result |
|---|---|
| Graphs store each feature at the expected length | yes: `hsv` 3, `ltp` 20, `ltp_rgb` 60, `ltp_hsv` 60, `gabor` 24, `gabor_rgb` 72, `gabor_hsv` 96 (uniform LTP with 8 neighbors has 20 bins per channel, not 118) |
| Each feature contributes nonzero evidence | yes, all 12 conditions: max single-feature evidence 1.00 for `hsv` and every LTP feature, 0.79-0.84 for Gabor. Gabor mean evidence is low (~0.02) at its placeholder tolerance 0.30 -- Phase 4 calibrates |
| **Combined-model equivalence** (gate for 1 training per dataset x lighting) | **pass**: combined vs `ltp_rgb`-only training store identical node positions and identical shared features for all 3 objects (946-982 nodes each); evals give identical target, result, most-likely object, steps and Monty steps per episode for `ltp_rgb` and the `hsv` baseline (3/3 each) |
| Determinism (same seed twice) | pass: identical per episode (3/3) |
| Step caps | eval episodes stop at `max_eval_steps` 500 (seen in shape-only runs); training episodes run `num_exploratory_steps` 1000. Walk-length overrides must also set `monty_args.num_exploratory_steps` (see 2026-10-04 log) |

## Compute budget (from Prelim C, 64-core machine)

Measured:
- Training episode (1000 exploratory steps): **~584 s with the combined
  sensor module** vs ~177 s `ltp_rgb`-only (3 concurrent processes). The
  combined module is ~3.3x slower per episode but replaces ~7 per-feature
  trainings.
- Eval: **~50-56 ms per Monty step**, flat between 2.9k and 27k graph nodes
  (3-object model; 15-object Phase 2 model). Episodes took a median 127 Monty
  steps (3 objects) and 1,009 (15 objects); a full 14-rotation model holds
  ~8k nodes per object (~400-480k nodes), beyond the measured range.

Arithmetic:
- Training: 2 datasets x 2 lightings x (50 or 60) objects x 14 rotations =
  1,540 episodes x ~584 s = ~250 CPU-h. `run_parallel.py` runs one process per
  object (its 14 rotations in sequence, ~2.3 h); 220 object-processes over 64
  cores = ~3.4 rounds -> **~8 h wall** for all four trainings.
- Phase 5 evals: 8 conditions x 110 objects x 5 rotations x 2 seeds = 8,800
  episodes per lighting. Assuming ~1,500 Monty steps x 55 ms = ~80 s per
  episode: ~200 CPU-h per lighting -> **~3.3 h wall per lighting** on 60
  processes, ~6.5 h for both.
- Phase 3 extras (`ltp_hsv` arms, 2 rotations x 1 seed): < 1 h.

Decision: keep **5 rotations x 2 seeds** (provisional) and re-measure on the
first ~50 real eval episodes; if they average <= 60 s, raise to 10 rotations
x 2 seeds (+~6.5 h). The per-step cost at 400k+ nodes is the main uncertainty.

## Log

Newest first. One or two lines each: what ran, result, where the output is.

- **2026-10-05** -- Main trainings launched. Step 0 check of the resolved
  configs: sensor module emits `hsv`, `ltp`, `ltp_rgb`, `ltp_hsv`, `gabor`,
  `gabor_rgb`, `gabor_hsv`; `lights` = `AMBIENT_LIGHTS` (`_amb`) / `null`
  (`_dir`); data `compat3d_texture` (50 objects) / `compat3d_mixed` (60);
  14 predefined rotations, 1000 exploratory steps. Driver
  `~/tbp/results/monty/journal_logs/run_main_trainings.sh` runs the ambient
  pair concurrently (`run_parallel.py`, 25 + 30 processes = 2 rounds each),
  then the directional pair; progress in `main_trainings.status`.
  First launch aborted after 7 min: load average ~1,700 on 64 cores because
  every process opened a full BLAS/OpenMP thread pool. Relaunched 13:16 with
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`: load ~85. (The
  Prelim C timings used 3 processes and unrestricted threads, so the ~8 h
  estimate is re-measured on this run.)
  Measured over the first 30 min: 119 episodes finished on 55 processes ->
  ~830 s per training episode at full load (1.4x Prelim C). Each lighting
  is 2 rounds x 14 episodes -> ~6.5 h; ambient expected ~19:45, directional
  ~02:15 on 2026-10-06 (~13 h, not ~8 h).
  Update 14:47: the second 57 min finished 155 episodes (~1,220 s each);
  episodes slow as each object's graph grows over its 14 rotations. Revised:
  ambient ~22:30, directional ~08:00 on 2026-10-06.
- **2026-10-07** -- Decisions (user): keep 3DC-Texture despite `hsv` 85%;
  test the speed claim on 3DC-Mixed only. Defaults applied: tolerances from
  the all-other-objects rule; Phase 5 at 5 rotations x 2 seeds. Plan updated
  (claim, Prelim B outcome, Phase 4 rule, Phase 5 design and metrics).
  Session wrapped up; Phase 5 starts in a new chat (see "Next session").
- **2026-10-06 10:01** -- Directional trainings done (Texture 700, Mixed
  840 episodes, no errors, 9.0 h). Phase 4 run on all four models (ambient
  rows identical); 40 eval configs regenerated with each model's own
  tolerance; snapshots updated, conf tests pass (170).
- **2026-10-06 01:20** -- Phase 3 done: `ltp_rgb` chosen (section above).
  The 3DC-Texture `ltp` run hit the same crash in episodes 24-27 (it started
  before the fix); rerun and merged by hand. Directional trainings started
  00:49 (driver `run_prelim_evals_then_dir_v2.sh`), expected ~10:30.
- **2026-10-06 00:30** -- Prelim B done (section above). Its 3DC-Texture
  shape-only run aborted on a crash: `ValueError` from a complex surface
  normal. `surface_normal_total_least_squares` uses `np.linalg.eig` on a
  symmetric covariance matrix, and round-off returned a complex pair when two
  eigenvalues were nearly equal (same pattern in `curvature_at_point` and
  `principal_curvatures`). Fix: take real parts (renormalize the normal) *only*
  when the result is complex, so every case that did not crash is
  bit-identical and episodes pair across runs made before and after the fix;
  2 regression tests (fail without the fix). `run_parallel` aborts the whole
  run on one worker exception, so episodes 24-27 were rerun and merged by
  hand into that run's `eval_stats.csv` (note in the run dir).
- **2026-10-05 23:38** -- Prelim evals relaunched (3rd try) at 8 + 12
  processes with a memory watchdog (`journal_logs/run_prelim_evals_then_dir_v2.sh`);
  memory steady at 62-84 of 123 GB. What went wrong before:
  1. 22:42: the kernel's out-of-memory killer killed 8 workers; their pool
     then hung (Python's `multiprocessing.Pool` waits forever for a dead
     worker). Cause: every eval episode saved a full copy of the ~0.9 GB model
     (`MontyExperiment.post_epoch` -> `save_state_dir`; upstream skips it only
     under parallel wandb logging), so each process peaked at 8.4 GB, and
     212 episodes wrote 241 GB to disk. **Fix:** new experiment config key
     `save_eval_models` (default `true`, upstream behavior unchanged); the
     journal eval configs set it `false` (generator + snapshots updated; conf
     and experiment tests pass, 174). Peak per process 8.4 -> ~3 GB; the
     duplicate model files were deleted.
  2. 23:33: at 16 + 20 processes memory still reached 121 of 123 GB: under
     load workers hold 3.4-5 GB (one worker alone peaks at 2.75 GB; not yet
     explained). Stopped before any kill. Hence 8 + 12.
  Eval episode cost (ambient, real models): 3DC-Mixed shape-only median
  36 s; 3DC-Texture shape-only ~95 s (500-step cap common). **Phase 5
  budget consequence:** memory, not cores, caps eval parallelism at ~20
  processes, so 8,800 episodes per lighting at ~60 s average is ~7 h wall
  per lighting (the Prelim C plan assumed 60 processes, ~3.3 h).
- **2026-10-05 22:41** -- Ambient trainings done (Texture 700, Mixed 840
  episodes, no errors; 8.8 / 9.0 h). Phase 4 (ambient) done (section above);
  20 eval configs regenerated with the calibrated tolerances. Prelim B /
  Phase 3 evals started 22:41. Fixes on the way: (1) the tolerance sweep over
  ~100M distance pairs ran for hours on one core -> distances subsampled to
  2M pairs per regime (`--max-pairs`): the same Phase 2 result (0.256 vs
  0.255) in 20 s; (2) `collect_distances` counted each query's zero distance
  to itself in `within_group` (negligible on 3DC-Texture; on 3DC-Mixed,
  where no shape repeats, it was the *only* entry) -- fixed, with unit tests.
- **2026-10-05 15:01** -- Reordered the queue: after the ambient trainings,
  Phase 4 (ambient) calibration and the Prelim B / Phase 3 evals run on the
  whole machine *before* the directional trainings (prelim gates report
  ~00:30 instead of next midday; directional slips ~2 h, to ~10:00). The old
  driver's bash process was stopped (the ambient jobs kept running) and
  replaced by `journal_logs/run_after_ambient.sh`. Prelim B / Phase 3 runs are
  named `prelim_eval_..._2rot` and write to `~/tbp/results/monty/projects/journal`.
  Their 2 rotations are the first 2 of Phase 5's 5: eval rotations come from
  `episode_seed(seed, mode, episode)`, so episodes pair across phases.
- **2026-10-05** -- `generate_journal_configs.py` now reads Phase 4's CSV and
  overrides each eval's texture tolerance with its own model's calibrated
  value (`TOLERANCE_REGIME`, default `across_object`); checked that the
  override resolves. Without the CSV the output is unchanged.
- **2026-10-05** -- Parallel eval path checked end to end (`run_parallel.py`,
  2 episodes, Phase 2 model): `eval_stats.csv` written with `episode_seed`.
  **Budget risk:** both episodes hit the 500-step cap and took ~280 s each
  on the loaded machine, vs the ~80 s Prelim C budget assumed. If real
  episodes are similar, Phase 5 at 5 x 2 is ~12 h wall per lighting, not
  ~3.3 h. Re-measure on the Prelim B episodes.
- **2026-10-05** -- `scripts/phase4_calibration.py` written; added
  `normalize=False` to `extract_histograms` so Gabor vectors are not
  renormalized like histograms. Open choice for Phase 4: which "other object"
  regime picks the tolerance -- `across_object` (plan text) or `same_shape`
  (the 3DC-Texture confusers, as in Phase 2); both are reported.
- **2026-10-05** -- `scripts/phase5_stats.py` written (reuses
  `pvalue_test.py`'s exact McNemar). On synthetic runs with planted effects it
  recovers them: proposed - baseline +15.4 points (planted +15) ambient,
  +25.0 (planted +25) directional, interaction +9.6 (planted +10).
- **2026-10-05** -- Phase 6 per-step timing harness
  `scripts/phase6_step_timing.py`: wraps `MontyBase.step` and
  `motor_only_step` (together = every Monty step), `CameraSM.step`,
  `ObservationProcessor.process` (feature extraction) and
  `GraphLM.matching_step`; one JSON line per episode. Smoke test (Phase 2
  0.12 m model, 1 jug episode, `ltp_rgb`, machine loaded by the trainings, so
  not a measurement): 1,937 Monty steps = `monty_steps` in eval stats; 500
  call `step` (LM runs), 1,437 `motor_only_step`; the model accounts for
  ~99 s of a 211 s episode, the simulator and logging for the rest.

- **2026-10-05** -- Prelim C done, all gates pass (section above). Main
  configs generated by `scripts/generate_journal_configs.py`: 12 conditions
  (`learning_module/journal/main/`), per-texture eval sensor modules,
  `env_interface/journal/{train,eval}_3dc_{texture,mixed}`, and 52 experiments
  (`experiment/journal/pretrain_3dc_*_all_features`, `eval_3dc_<dataset>_<lighting>_<condition>`).
  Budget: ~8 h for the 4 trainings, ~6.5 h for Phase 5 evals at 5 x 2.

- **2026-10-04** -- Datasets frozen. `select_dataset_shapes.py` built and
  walked ranked candidates (`make_recipes.py --candidates`) per class and kept
  the first passing; recipes overwritten with the frozen selection (header
  says do not regenerate) plus `selection_log.csv`. Texture: dresser and
  cabinet failed 30 candidates each -> love_seat, stool. Rebuilt both sets in
  `$MONTY_DATA` from the frozen recipes; final Prelim A clean. Gabor wired into
  CameraSM / calculator / graph code; combined-feature SM config
  `monty/sensor_module/journal/camera_surf_rgba_all_texture.yaml`.

- **2026-10-04** -- Built `compat3d_texture` and `compat3d_mixed` at 0.12 m;
  ran Prelim A (results section above): static checks pass, on-object and
  slot-visit checks fail for many shapes. Found why `max_total_steps=200`
  didn't cap episodes: `monty_experiment.py:187` resets it to
  `num_exploratory_steps + max_train_steps` when below `num_exploratory_steps`
  (1000), so walks must also set `monty_args.num_exploratory_steps`. Gabor
  extraction functions written (`gabor`, `gabor_rgb`, `gabor_hsv`; FFT and
  spatial paths agree to 1e-5); rough timing 22 ms/patch (FFT) vs `ltp_rgb`
  4 ms.

- **2026-10-04** -- `target_size` stays **0.12 m**: user judged the Phase 2
  margin for 0.25 m not significant; recorded, revisit if needed. Phase 7's
  0.5x set (0.06 m) needs building.
- **2026-10-04** -- Phase 2 ran. Built `compat3d_texture_p2_{012,025,050}`;
  pretrained `journal/phase2_pre_training_texture_p2_{012,025,050}_amb`
  (ambient, 2 rotations, `run_parallel.py`, ~11 min each). Analysis
  (`scripts/phase2_footprint_analysis.py`, new `location_specificity_auc`)
  selects **0.25 m** by the pre-registered rule; table above. New configs:
  `environment/habitat_3dcompat_journal_surf_agent_{amb,dir}.yaml` (explicit
  `lights`), `env_interface/train_3dcompat_texture_p2_2rot.yaml`; snapshots
  added; conf tests 118 pass. Phase 2 code not yet committed.

- **2026-10-04** -- Phase 2 cut from 14 to 2 rotations per object
  (`[0,0,0]`, `[90,0,0]`): its two measures are offline statistics on stored
  features, not a full model. 630 -> 90 episodes, ~10-15 min wall time
  (estimate). User approved.

- **2026-10-04** -- Plan restructured for time, at the user's request.
  Ambient is the primary scenario for every phase; directional is the second
  scenario of Phase 5 (same conditions/objects/rotations/seeds, interaction
  CI), not a separate late phase (old Phase 8 removed). Time savings: one
  combined-feature training per dataset x lighting (4 instead of ~20; gated by
  a Prelim C equivalence check), all builds on day 1 at all three sizes,
  Gabor moved to days 1-2, Phase 3 / Prelim B reuse the main models,
  directional runs queued right behind ambient. Schedule ~12 working days
  (was ~20).
- **2026-10-04** -- (superseded) Added a lighting experiment as Phase 8.

- **2026-10-04** -- User approved continuing with the revised 3DC-Texture
  recipe. Phase 2 subset changed from "first 3 shapes" (all wood) to one shape
  per family: jug (metal), ottoman (leather), table (wood). Plan updated
  (dataset design, build tooling, Prelim A brightness + slot-visit checks,
  Phase 2 measured footprint, risks). Open issues: (1) dark materials render
  nearly black under default (= directional) lighting -- re-check under ambient,
  the primary scenario (table wood_13, chair leathers, bed
  fabric_27) -- brightness floor TBD in Prelim A; (2) the sofa's varied slot
  (back + bottom panels) is hidden from the front; (3) a
  `max_total_steps=200` override did not stop pretraining episodes (~1,200
  steps) -- find how the cap applies before Prelim C timing.
- **2026-10-04** -- What the sensor sees, checked with real surface-agent
  pretraining episodes (`show_sensor_output` frames, 10 objects; raw arrays
  from one ottoman episode). Patch 2.5 cm from the surface, 9° field of view
  -> 3.9 mm footprint, ~2.3 px of the 64-px view finder (the plotter's white
  box draws 6.4 px). One `ltp_rgb` histogram per step over the whole 64x64
  patch (on-object pixels); the view finder feeds no features. The earlier
  preview used a distant agent and was misleading. Capture script saved as
  `scripts/compat3d_test/capture_sensor_output.py`.
- **2026-10-04** -- User rejected the one-material-per-object 3DC-Texture design
  (unrealistic objects). Redesigned: each shape starts from a dataset
  composition; the 4 variants swap the material of one slot (the parts that
  share the largest visible material) for other same-family materials closest
  in mean color. "Visible" = point-splat visibility from 42 views, since raw
  area counted hidden inner panels. 3DC-Mixed unchanged (already composition 0).
- **2026-10-04** -- `material_stats.py` and `make_recipes.py` written in
  `~/3dcompat-habitat-prep` (uncommitted). 293 materials measured; 8 flagged as
  near-duplicate images of another material and excluded. First recipes
  (one material per object, hue window ±15°, texture-energy floor 5) superseded
  by the entry above.

- **2026-10-04** -- Journal configs moved to `experiment/journal/` and renamed
  after their feature keys (old `randrot_3dcompat_ltp_hsv` was the center-pixel
  `hsv` baseline, now `randrot_3dcompat11_hsv`). Texture weights set to 1.0 in
  new `learning_module/journal/` configs (all LTP arms were 2.0). Any models
  pretrained under the old names must be retrained (new run names). Conf tests
  pass (115).
- **2026-10-04** -- Tracker created. Nothing from the plan started; inventory
  of existing pieces above.
