# Progress log: journal experiment plan

Tracks completion of [JOURNAL_EXPERIMENT_PLAN.md](JOURNAL_EXPERIMENT_PLAN.md).
Update the status table and add a dated log entry whenever something finishes,
fails, or a decision is made. Numbers go here only once regenerated under the
plan (prior BICA-era results are ignored).

Status legend: **done** · **in progress** · **todo** · **blocked** · **n/a**

Last updated: 2026-10-04

## At a glance

| Phase | Status | Gate / output | Note |
|---|---|---|---|
| Build tooling (`material_stats.py`, `make_recipes.py`) | done | `material_stats.csv`, recipe YAMLs | 3DC-Texture design approved by user 2026-10-04 |
| Phase 2 -- scale / footprint sweep (ambient) | done | `target_size` decision | **0.12 m kept** (user decision); rule would pick 0.25 m by a small margin. Results below |
| Datasets 3DC-Texture (50) + 3DC-Mixed (60) | **done, frozen** | `$MONTY_DATA/compat3d_texture`, `compat3d_mixed` | Shapes chosen once by `select_dataset_shapes.py` (Prelim A walk rule); recipes frozen in `~/3dcompat-habitat-prep/recipes/` |
| Phase 1 Prelim A -- asset sanity | **done** | 100% pass on 5 checks + contact sheet | Passed via selection: every frozen object meets the walk rule; final static checks clean |
| Phase 1 Prelim B -- headroom | todo | `hsv` baseline in 30-80% on 3DC-Texture | |
| Phase 1 Prelim C -- smoke + compute budget | todo | budget arithmetic written into this doc | Also gates combined-feature training and checks step caps |
| Phase 3 -- color space pilot (`ltp_rgb` vs `ltp_hsv`, ambient) | todo | decision record | Evals on the main models; no longer gates training |
| Phase 4 -- tolerance calibration | todo | tolerance per feature, per lighting | |
| Gabor feature (`gabor`, `gabor_rgb`, `gabor_hsv`) | done | functions + unit tests + FFT path | Extraction, CameraSM, evidence calculator (RMS of per-channel Euclidean), excluded from node-keeping; 21 new tests; full unit suite 668 pass |
| Phase 5 -- main comparison: ambient (primary), then directional | todo | accuracy/steps table per scenario, McNemar/Wilcoxon, interaction CI | 4 combined-feature trainings (2 datasets x 2 lightings) |
| Phase 6 -- wall time | todo | per-patch µs, per-step, time-to-correct | No timing harness yet |
| Phase 7 -- robustness (ambient) | todo | noise / footprint mismatch / threshold | Lighting is a Phase 5 scenario, not an ablation |
| Stats, figures, writing | todo | | |

## Pre-existing pieces (checked 2026-10-04)

| Item | State | Where |
|---|---|---|
| Feature keys `ltp`, `ltp_rgb`, `ltp_hsv` | present | `frameworks/utils/sensor_processing.py`, `frameworks/models/sensor_modules.py` |
| Offline distance analysis (`tolerance_sweep`, `best_tolerance`, Hellinger + 6 others) | present | `frameworks/utils/ltp_distance_analysis.py` |
| Location-specificity AUC function (Phase 2 measure 2) | present, uncommitted (+ unit tests) | `ltp_distance_analysis.location_specificity_auc`; analysis script `scripts/phase2_footprint_analysis.py` |
| `compat3d_helper.py --recipe ... --build` | present | `~/3dcompat-habitat-prep/` |
| Source zip `3DCoMPaT_ZIP.zip` | present | `~/3dcompat-habitat-prep/3DCoMPaT_3D/` |
| Old 11-object sets `compat3d_ltp`, `compat3d_habitat_test` | present | `~/tbp/data/` (= `$MONTY_DATA`, exported in `~/.bashrc`) |
| Env / train / eval configs for `compat3d_ltp` | present, uncommitted | `conf/env_interface/*3dcompat*`, `conf/environment/habitat_3dcompat_*` |
| Sensor-patch renderer (distant agent at 0.3 m -- *not* the surface-agent experiments' view) | present | `scripts/compat3d_test/render_sensor_snapshots.py` |
| Trusted sensor-view capture (`show_sensor_output` frames, headless) | present | `scripts/compat3d_test/capture_sensor_output.py` |
| Material stats + recipe generator | present (committed on branch `journal-datasets`) | `~/3dcompat-habitat-prep/{material_stats.py,make_recipes.py,material_stats.csv,recipes/}` |
| Configs for 3DC-Texture / 3DC-Mixed | **missing** | |
| Ambient light setup `AMBIENT_LIGHTS`; directional = Habitat default (`lights: None`) | present | `simulators/habitat/simulator.py` |
| Ambient and directional 3DCoMPaT environment / experiment configs for the new sets (explicit `lights` in both) | **missing** | template: `conf/environment/retextured_surf_agent_semantics0_amb.yaml`. All current 3DCoMPaT configs set no `lights` key, so they run directional |
| Gabor feature | **missing** | |
| Timing harness | **missing** | |

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
| Color LTP winner (`ltp_rgb` or `ltp_hsv`) | -- | | Phase 3 rule: `ltp_hsv` only if ≥ 3 points better on both sets |
| Tolerances per feature | -- | | Phase 4 |
| Dataset shape selection rule | 200-step ambient walks from [0,0,0] and [90,0,0]: on-object > 0.9 (mean); 3DC-Texture also slot visits >= 0.4 and every variant's slot brightness >= 0.15 | 2026-10-04 | User-approved; run once, recipes frozen. Log: `selection_log.csv` (prep `recipes/`, copy in `docs/ltp/results/prelim_a/`) |
| 3DC-Texture classes | table, love_seat, stool (wood); sofa, bed (fabric); chair, ottoman (leather); vase, jug (metal); toilet (marble) | 2026-10-04 | dresser and cabinet failed all 30 candidates each; user chose to swap in classes; love_seat and stool were the first two passing (garbage_bin, bag also passed, not needed) |
| 3DC-Mixed classes | 28 classes, 60 shapes | 2026-10-04 | shelf and skateboard had no passing shape in 20; cabinet, dishwasher, dresser 1 each; 7 classes topped up to 3 to keep 60 |
| Primary lighting scenario | ambient (`AMBIENT_LIGHTS`); directional is the second scenario | 2026-10-04 | User decision: ambient is the default all phases run in |
| Tolerances per feature, directional | -- | | Phase 4 rule on directional training graphs; directional models also evaluated at ambient tolerances |
| Directional-scenario conditions | all 8 Phase 5 conditions (provisional) | | Cut to baseline / proposed / `ltp` / Gabor matched if the Prelim C budget requires |
| Trainings | 1 combined-feature run per dataset x lighting (4 total) | 2026-10-04 | Texture features don't enter `graph_delta_thresholds`; gate: Prelim C equivalence check |
| Rotations x seeds for Phase 5 | 5 x 2 (provisional) | | Prelim C budget |

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

## Compute budget (fill from Prelim C)

- s / training episode (50-obj, 60-obj): --
- s / eval episode (50-obj, 60-obj): --
- Arithmetic: --

## Log

Newest first. One or two lines each: what ran, result, where the output is.

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
