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
| Build tooling (`material_stats.py`, `make_recipes.py`) | done (uncommitted) | `material_stats.csv`, recipe YAMLs | 3DC-Texture design approved by user 2026-10-04 |
| Phase 2 -- scale / footprint sweep (ambient) | todo (next) | `target_size` decision | Subset recipes ready: jug, ottoman, table x 5 variants at 0.12 / 0.25 / 0.5 m; 2 rotations per object (90 episodes) |
| Datasets 3DC-Texture (50) + 3DC-Mixed (60) | todo | `$MONTY_DATA/compat3d_texture`, `compat3d_mixed` | Build at all 3 sizes on day 1 (no longer waits for Phase 2) |
| Phase 1 Prelim A -- asset sanity | todo | 100% pass on 5 checks + contact sheet | |
| Phase 1 Prelim B -- headroom | todo | `hsv` baseline in 30-80% on 3DC-Texture | |
| Phase 1 Prelim C -- smoke + compute budget | todo | budget arithmetic written into this doc | Also gates combined-feature training and checks step caps |
| Phase 3 -- color space pilot (`ltp_rgb` vs `ltp_hsv`, ambient) | todo | decision record | Evals on the main models; no longer gates training |
| Phase 4 -- tolerance calibration | todo | tolerance per feature, per lighting | |
| Gabor feature (`gabor`, `gabor_rgb`, `gabor_hsv`) | todo | functions + unit tests + FFT path | **On the critical path** (days 1-2): the main trainings store it |
| Phase 5 -- main comparison: ambient (primary), then directional | todo | accuracy/steps table per scenario, McNemar/Wilcoxon, interaction CI | 4 combined-feature trainings (2 datasets x 2 lightings) |
| Phase 6 -- wall time | todo | per-patch µs, per-step, time-to-correct | No timing harness yet |
| Phase 7 -- robustness (ambient) | todo | noise / footprint mismatch / threshold | Lighting is a Phase 5 scenario, not an ablation |
| Stats, figures, writing | todo | | |

## Pre-existing pieces (checked 2026-10-04)

| Item | State | Where |
|---|---|---|
| Feature keys `ltp`, `ltp_rgb`, `ltp_hsv` | present | `frameworks/utils/sensor_processing.py`, `frameworks/models/sensor_modules.py` |
| Offline distance analysis (`tolerance_sweep`, `best_tolerance`, Hellinger + 6 others) | present | `frameworks/utils/ltp_distance_analysis.py` |
| Location-specificity AUC function (Phase 2 measure 2) | **missing** | to add in `ltp_distance_analysis.py` |
| `compat3d_helper.py --recipe ... --build` | present | `~/3dcompat-habitat-prep/` |
| Source zip `3DCoMPaT_ZIP.zip` | present | `~/3dcompat-habitat-prep/3DCoMPaT_3D/` |
| Old 11-object sets `compat3d_ltp`, `compat3d_habitat_test` | present | `~/tbp/data/` (= `$MONTY_DATA`, exported in `~/.bashrc`) |
| Env / train / eval configs for `compat3d_ltp` | present, uncommitted | `conf/env_interface/*3dcompat*`, `conf/environment/habitat_3dcompat_*` |
| Sensor-patch renderer (distant agent at 0.3 m -- *not* the surface-agent experiments' view) | present | `scripts/compat3d_test/render_sensor_snapshots.py` |
| Trusted sensor-view capture (`show_sensor_output` frames, headless) | present, uncommitted | `scripts/compat3d_test/capture_sensor_output.py` |
| Material stats + recipe generator | present, uncommitted | `~/3dcompat-habitat-prep/{material_stats.py,make_recipes.py,material_stats.csv,recipes/}` |
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
| `target_size` (patch footprint) | -- | | Phase 2 |
| 3DC-Texture design | dataset composition + 4 same-family swaps of the most visible slot (>= 40% of visible surface), closest in mean color, texture energy >= 5 | 2026-10-04 | User rejected one-material-per-object (unrealistic); approved this after Habitat previews and `show_sensor_output` frames |
| Patch footprint at `target_size` 0.12 | 3.9 mm of surface (~3% of object) | 2026-10-04 | Raw sensor arrays from a real ottoman episode; judged usable by user. Phase 2 still decides `target_size` |
| Color LTP winner (`ltp_rgb` or `ltp_hsv`) | -- | | Phase 3 rule: `ltp_hsv` only if ≥ 3 points better on both sets |
| Tolerances per feature | -- | | Phase 4 |
| Primary lighting scenario | ambient (`AMBIENT_LIGHTS`); directional is the second scenario | 2026-10-04 | User decision: ambient is the default all phases run in |
| Tolerances per feature, directional | -- | | Phase 4 rule on directional training graphs; directional models also evaluated at ambient tolerances |
| Directional-scenario conditions | all 8 Phase 5 conditions (provisional) | | Cut to baseline / proposed / `ltp` / Gabor matched if the Prelim C budget requires |
| Trainings | 1 combined-feature run per dataset x lighting (4 total) | 2026-10-04 | Texture features don't enter `graph_delta_thresholds`; gate: Prelim C equivalence check |
| Rotations x seeds for Phase 5 | 5 x 2 (provisional) | | Prelim C budget |

## Compute budget (fill from Prelim C)

- s / training episode (50-obj, 60-obj): --
- s / eval episode (50-obj, 60-obj): --
- Arithmetic: --

## Log

Newest first. One or two lines each: what ran, result, where the output is.

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
