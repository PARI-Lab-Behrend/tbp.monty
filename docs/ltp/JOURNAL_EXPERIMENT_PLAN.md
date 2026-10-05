# Experiment plan: color LTP for Monty (Cognitive Systems Research)

## Claim the paper must support

Adding a color Local Ternary Pattern feature (LTP: a histogram of per-pixel
"brighter / darker than the center by more than a threshold" codes over the sensed
patch) to Monty's sensor module makes object recognition **more accurate** and
**faster in steps**, on same-shape/different-texture objects from 3DCoMPaT++, and
does so at **lower wall-clock cost than a Gabor-filter texture feature** of
comparable discriminative power.

Each claim maps to one table or figure:

| Claim | Evidence | Section |
|---|---|---|
| Color LTP beats Monty's default (center-pixel hue `hsv`) | Accuracy + steps on two 3DCoMPaT++ sets | Phase 5 |
| Color matters (not only texture) | Grayscale `ltp` vs color LTP | Phase 5 |
| RGB vs HSV channel choice is justified | Pilot decision record | Phase 3 |
| LTP is cheaper than Gabor | Per-patch µs and per-episode wall time | Phase 6 |
| The 3DCoMPaT++ scale/view problem is understood and controlled | Patch-footprint sweep | Phase 2 |
| The advantage holds under directional lighting, not only the ambient default | Phase 5 conditions under both lighting scenarios | Phase 5 |
| Robust beyond the tuned setting | Noise, train/test scale mismatch, LTP threshold | Phase 7 |

Naming note: the default Monty color feature is the center-pixel `hsv` feature.
LTP computed separately over the H, S and V channels is the `ltp_hsv` feature.
These are different things; the plan always uses the feature key, never "HSV
LTP" or "LTP-HSV", to avoid confusing them.

## What already exists (reuse, do not rebuild)

- Color LTP extraction: `get_ltp_rgb_texture_feature_vector` and
  `get_ltp_hsv_texture_feature_vector` in
  `src/tbp/monty/frameworks/utils/sensor_processing.py` (feature keys `ltp_rgb`,
  `ltp_hsv`; grayscale `ltp`).
- Offline distance/tolerance calibration from training graphs:
  `src/tbp/monty/frameworks/utils/ltp_distance_analysis.py`
  (`tolerance_sweep`, `best_tolerance`, Hellinger and 6 other metrics).
- 3DCoMPaT++ to Habitat converter: `~/3dcompat-habitat-prep/convert_compat3d_to_habitat.py`
  (`--target-size` = max object extent in meters, default 0.12).
- Prepared sets in `$MONTY_DATA`: `compat3d_ltp` (ottoman x4 leather, bench x4
  wood, chair/table/boat controls) and `compat3d_habitat_test`.
- Env / train / eval configs for `compat3d_ltp`; `run_parallel.py` for parallel
  episodes; noise transform `missing_noise_depthto3d_sensor2_semantic0`.
- Trusted sensor view: `scripts/compat3d_test/capture_sensor_output.py` runs any
  experiment with `show_sensor_output=true` headless and saves the live-plotter
  frames plus the patch RGB. Use this, not `render_sensor_snapshots.py` (a
  *distant* agent 0.3 m away, which is not what the surface-agent experiments
  see), whenever a claim depends on what the sensor sees.

**Not existing:** a Gabor feature; a timing harness; a scale sweep. Prior run
results are ignored per instruction -- every number in the paper is regenerated.

**Step 0 rule:** before any run, verify each config's *resolved* contents
(`python run.py experiment=<name> --cfg job`), not its file name: which features
the sensor module emits, which have nonzero LM weights, the tolerance, dataset
path, `target_size` the objects were built with, and `env_init_args.lights`.
Record these in the results table.

## Lighting scenarios (a design factor in every phase)

Two scenarios, set by the `lights` argument of `HabitatEnvironment`. Verify the
*resolved* `lights` of every run, not the config name.

| Scenario | Role | Resolved `lights` | What it is |
|---|---|---|---|
| **ambient** | **primary** -- every phase runs here; all decisions (Phase 2-4) are made here | `AMBIENT_LIGHTS` (`simulators/habitat/simulator.py`) | six equal directional lights along ±x, ±y, ±z, calibrated to the default's mean brightness; mild shading |
| **directional** | **second scenario** -- Phase 4 calibration and the Phase 5 comparison are repeated here | `None` (Habitat's built-in setup) | five directional lights weighted toward front and above; strong pose-dependent shading that shifts the center-pixel `hsv` value and adds intensity gradients across the patch |

Every existing 3DCoMPaT config sets no `lights` key and therefore runs
**directional**. The new journal configs must set `lights` explicitly in both
scenarios (template: `conf/environment/retextured_surf_agent_semantics0_amb.yaml`)
so no run inherits a scenario by omission. Run names carry `_amb` / `_dir`.

## Train once per dataset x lighting (time saving; verified in Prelim C)

Pretraining keeps or drops graph nodes using only `distance`, `pose_vectors`,
`principal_curvatures_log` and `hsv` (`graph_delta_thresholds` in
`displacement_delta_d0001_ltp`); texture features do not affect which nodes
are stored, and the existing `hsv` baseline eval already reuses the grayscale
`ltp` model. So **one pretraining run whose sensor module emits every feature**
(`hsv`, `ltp`, `ltp_rgb`, `ltp_hsv`, `gabor`, `gabor_rgb`) produces the same
graphs as one run per feature set, and each condition is selected at eval time
by the learning module's `feature_weights` / `tolerances` alone.

- Trainings: 2 datasets x 2 lightings = **4**, instead of one per feature set
  per dataset per lighting (~20).
- Evals use a sensor module that emits only that condition's features, so
  per-step wall time (Phase 6) is measured fairly.
- Gate (Prelim C): on 3 objects, eval with the combined model must give
  identical steps and results to eval with a single-feature model, for
  `ltp_rgb` and the `hsv` baseline. If not, fall back to per-feature trainings.
- Consequence: Gabor must be implemented before the 4 main trainings start,
  so it moves onto the critical path (days 1-2).

## Datasets (two large 3DCoMPaT++ sets, built from recipes)

The existing 11-object sets (`compat3d_ltp`, `compat3d_habitat_test`) are too
small for a journal claim: 2 shapes means "texture helps" rests on two
geometries. Instead, build two sets of 50-60 objects, each answering a
different question.

What the source offers (counted from `3DCoMPaT_ZIP.zip`): 8,753 shapes in 42
classes; 294 materials in families (metal 66, wood 64, marble 39, leather 34,
fabric 32, plastic 21, granite 11, ceramic/glass 6 each, ...); up to 1,000
ready-made compositions (material-to-part assignments) per shape.

| Name in paper | Design | Objects | Question it answers |
|---|---|---|---|
| **3DC-Texture** | 10 shapes from 10 classes, each in one of the dataset's own compositions, x 5 variants that differ only in the material of the most visible *slot* (the parts sharing one material), swapped for other materials of the same family closest in mean color | 50 | Can texture separate objects when shape *and* color mostly cannot? (the hard case) |
| **3DC-Mixed** | 60 shapes over 28 classes (2 per class; 7 classes have 3 to make up for classes with fewer passing shapes), 1 dataset composition each (multi-material, as shipped) | 60 | Does LTP still help -- or at least not hurt -- in ordinary recognition where shape usually separates objects? |

Why these two: 3DC-Texture is the stress test (like YCB-11R but 5x larger and
with 10 geometries); 3DC-Mixed is the realism check (like YCB-77). A paper
that only wins the stress test invites "it only works on a contrived set".

Optional third set if time remains: 3DC-Texture-L, 20 shapes x 8 materials =
160 objects, to show accuracy holds as same-shape confusers grow.

### Build tooling (in `~/3dcompat-habitat-prep`, outside tbp.monty for licensing)

1. `material_stats.py` (done): for each of the 293 materials (excluding
   `placeholder`), load the diffuse texture from the zip and record mean H/S/V
   (hue weighted by saturation), hue spread, texture energy (std of the
   grayscale image; near zero means a flat color with nothing for LTP to see),
   and `duplicate_of` (8 materials ship a near-identical image under a second
   name). Output `material_stats.csv`.
2. `make_recipes.py` (done): writes the recipe YAMLs that
   `compat3d_helper.py --build` already consumes.
   - 3DC-Texture: for each (class, family) in `TEXTURE_PLAN` (dresser / table /
     cabinet: wood; sofa / bed: fabric; chair / ottoman: leather; vase / jug:
     metal; toilet: marble -- the families the dataset's compositions use on
     those classes), search shapes x compositions 0-19 for one whose most
     visible material slot is in that family and covers >= 40% of the visible
     surface. Visibility is a point-splat z-buffer from 42 directions, because
     raw surface area counted hidden inner shelves and back panels. The 4
     variants replace that slot's material with same-family materials that
     have texture energy >= 5, are not already on the object, and are closest
     in mean color (HSV cylinder distance). All other parts keep the
     composition's materials, so every object is realistic.
     (A first design -- one material over the whole object -- was rejected:
     the objects looked unrealistic.)
   - 3DC-Mixed: 2 random shapes per class (fixed seed), composition 0 of each.
   - Phase 2 subsets: one shape from each of 3 distinct families (currently
     jug / metal, ottoman / leather, table / wood), at each `target_size`.
   - With `--candidates N`: instead of picking the first qualifying
     candidate, write ranked candidates per class for step 2b.
2b. **Shape selection (run once, then frozen; done 2026-10-04).** The first
   recipes failed Prelim A: the surface agent lost many shapes, and on 4
   3DC-Texture shapes the patch never reached the varied slot.
   `scripts/compat3d_test/select_dataset_shapes.py` builds and walks each
   class's candidates in order (real 200-step ambient surface walks from
   [0,0,0] and [90,0,0]) and keeps the first that passes: on-object > 0.9;
   for 3DC-Texture also slot visits >= 0.4 (variants walk identical paths, so
   steps where base and variant patches differ are steps on the slot) and every
   variant's slot brightness >= 0.15. Its output overwrote the recipes, which
   are now the frozen dataset definition, plus `selection_log.csv` (every
   candidate and its rejection reason). Final 3DC-Texture classes: table,
   love_seat, stool (wood); sofa, bed (fabric); chair, ottoman (leather);
   vase, jug (metal); toilet (marble) -- dresser and cabinet replaced after
   failing 30 candidates each. Final 3DC-Mixed: 60 shapes over 28 classes
   (shelf and skateboard had no passing shape).
3. Build with `python compat3d_helper.py --zip ... --recipe <r>.yaml --build`.
   Building is cheap (~3.5 min per 50 objects), so build **both full sets at
   all three Phase 2 sizes (0.12, 0.25, 0.5 m) on day 1** instead of waiting
   for the Phase 2 decision; the chosen size is then already on disk.
   (Outcome: 0.12 m kept, so Phase 7's mismatch sets are 0.06 m -- to build --
   and 0.25 m.) Commit the recipes and `material_stats.csv` -- the
   recipe *is* the dataset definition.
4. Add env / train / eval configs in tbp.monty pointing at
   `$MONTY_DATA/compat3d_texture` and `$MONTY_DATA/compat3d_mixed`, one
   environment per lighting scenario.

## Phase 1 -- Build datasets + preliminary tests (days 1-4, overlapping other phases)

Three gating tests. Nothing in Phase 5 runs until all three pass on both sets.

### Prelim A -- Asset sanity (automated, ~1 h per set)

Catches broken objects before they become silent accuracy losses.

| Check | How | Pass |
|---|---|---|
| Loads in Habitat | spawn each object, one observation | 100% |
| Textures resolved | patch is not the `placeholder` material, not all-black, not one flat color (unless the material is flat) | 100% |
| Visible under both lighting scenarios | median on-object patch brightness (V) over a 200-step walk, under directional and under ambient light; dark materials (e.g. leathers with mean V <= 0.28) render nearly black under Habitat's default (directional) lights | threshold TBD from the first build; failures get a brighter replacement |
| Size | max extent == `target_size` ± 2% | 100% |
| Surface agent stays on object | 200-step surface walk per object; mean `on_object` | > 0.9 |
| Unique objects | no two objects build to the same name (helper hashes content) | 0 duplicates |

Also render a contact sheet from `scripts/compat3d_test/capture_sensor_output.py`
frames (view finder + patch RGB at several steps per object) and eyeball it
once, checking in particular that the patch actually visits the varied slot
(in the 10-object preview, the cabinet's patch spent steps on its unvaried door
and the vase's on its plant). Failed objects get replaced in the recipe, not
dropped silently.

### Prelim B -- Headroom (difficulty calibration, ~half day compute)

A dataset where Monty's default already gets ~100% cannot show LTP helping.
Runs on the ambient main models as soon as they finish training (no separate
training).

- Conditions: shape only; center-pixel `hsv` baseline. Ambient (primary);
  repeat under directional when those models finish, to know the headroom of
  the second scenario too.
- Design: every object x 2 random rotations x 1 seed, 500 max steps.
- Pass criteria:
  - 3DC-Texture: `hsv` baseline accuracy between 30% and 80%; shape only
    near chance among the 5 same-shape variants but correct on shape (i.e.
    confusions stay within the shape group). If `hsv` > 80%, replace the
    variants that `hsv` separates (the color-outliers in the shape group) or
    pick a different base composition, and rebuild.
  - 3DC-Mixed: shape only > 70% (shape mostly separates classes); `hsv` must
    not be at 100% *with zero* steps to spare -- check steps too.
- Output: confusion matrices; these also become a paper figure motivating
  the dataset.

### Prelim C -- Pipeline smoke test + compute budget (~half day)

Confirms every feature reaches the learning module and measures cost.

- One tiny combined-feature training (all feature keys) + one eval per
  feature key: 3 objects, 1 training rotation, 1 eval rotation, ambient.
  Runs at the chosen `target_size` on day 2 (or at 0.12 on day 1 if Gabor is
  not ready, then repeated for the budget).
- Pass criteria:
  - saved graphs contain the feature with the expected length (e.g. 3 x 20
    for `ltp_rgb` uniform with 8 neighbors);
  - the feature contributes nonzero evidence at least once (log per-feature
    evidence for one episode);
  - rerunning with the same seed gives identical steps and result
    (determinism, required for paired tests);
  - wall time per step logged (Phase 6 harness);
  - **combined-model equivalence**: eval with the combined model gives the
    same steps and result as eval with a single-feature model, for `ltp_rgb`
    and the `hsv` baseline (gate for training once per dataset x lighting);
  - `max_total_steps` / `max_train_steps` actually bound episode length (an
    override of `max_total_steps=200` did not, in the 2026-10-04 preview).
- Output: seconds per training episode and per eval episode for 50- and
  60-object models -> fixes rotations x seeds in Phase 5 to fit the compute
  budget (write the arithmetic into this doc).

## Phase 2 -- Scale / view problem (~half day: ~15 min compute + offline analysis; ambient)

**The problem.** The converter shrinks furniture (~1 m) to 0.12 m. The texture
shrinks with it, so a 64x64 patch at zoom 10 spans ~8x more of the material
pattern, and a larger fraction of the object, than on a native-size object.

**Measured at `target_size` 0.12** (real surface-agent episode, raw sensor
arrays, 2026-10-04): the patch sits 2.5 cm from the surface
(`desired_object_distance`) with a 9° field of view (Habitat zoom divides the
90° FOV), so it covers **~3.9 mm** of surface -- ~3% of the object, ~4-6 cm on
the full-size furniture. The view finder sits 3 cm further back and sees ~11 cm,
i.e. nearly the whole object, which makes objects *look* small; the patch is
~2.3 of its 64 px. The live plotter's white box (fixed 10% of the view finder)
overstates the patch ~3x in width. Patch RGB at this size shows readable
texture (leather grain, diamond plate, fabric weave, marble veins). So the
failure modes below are a risk to measure, not an observed problem:

- *Texture aliasing*: the material pattern is finer than a pixel, so LTP sees
  blur/noise, not texture.
- *Low location-specificity*: every patch sees "most of the object", so the
  feature is nearly the same everywhere on the object. It still separates
  objects, but adds little to Monty's pose hypotheses (features at pose).

**Independent variable: patch footprint on the surface** (mm of surface per
patch), set by either object `target_size` {0.12, 0.25, 0.5 m} or sensor zoom
{5, 10, 20}. Prefer `target_size`: it keeps the sensor identical to the YCB
setup Monty is tuned for.

Run on a 15-object subset of 3DC-Texture (3 shapes from 3 different families
x 5 variants), built at each size from the same recipe with only
`target_size` changed (`recipes/compat3d_texture_p2_{012,025,050}.yaml`). The
three subset trainings run in parallel under ambient light, **2 rotations
per object** (90 episodes in all, ~10-15 min wall time; Prelim C gives the real
number). Phase 2 gates the
main trainings, not the builds: full sets are built at all three sizes on day 1.

**Two offline measures per footprint**, from training graphs only (1 train
run per footprint, `ltp_rgb`, ambient, no eval). "Training" here only means
walking the surface agent over each object and storing features at locations;
both measures need that store, not a full model. So each run uses 2 rotations,
`[0, 0, 0]` and `[90, 0, 0]` (side, then pitched to bring the top or bottom
into view), via `object_init_sampler.rotations` with `n_train_epochs=2`,
instead of all 14 in `rotations_all`. If a measure is noisy (e.g. too few node
pairs < 1 cm or > 5 cm apart), add rotations rather than objects.

1. *Object separability*: margin between within-object and across-object
   Hellinger distances (existing `tolerance_sweep`). Restrict "across" to the
   same-shape confusers -- that is the hard case.
2. *Location-specificity*: within one object, AUC of feature distance for
   spatially near node pairs (<1 cm) vs far pairs (>5 cm). AUC ~0.5 means the
   feature carries no location information. New ~40-line function in
   `ltp_distance_analysis.py`.

**Decision:** pick the footprint that maximizes separability subject to
location-specificity AUC > 0.6; fix it for all main experiments.
**Outcome (2026-10-04):** the rule favored 0.25 m by a small margin (evidence
gap 0.353 vs 0.312 at 0.12 m; one training per size); the margin was judged
not significant and the default **0.12 m** was kept. Results in
JOURNAL_PROGRESS.md. Report the
sweep as a figure -- it is a contribution in itself (the dataset property that
YCB does not stress). Phase 7 then tests train/test footprint mismatch.

## Phase 3 -- Preliminary: which color space (0.5 day; ambient)

Uses the ambient main models (they store every LTP variant), so Phase 3 adds
evals only and no longer gates training. Conditions, all with `hsv` weight 0 so
the LTP variant is the only non-shape feature:

| Channels LTP is computed on | Feature key |
|---|---|
| none (shape only) | -- |
| grayscale | `ltp` |
| R, G, B | `ltp_rgb` |
| H, S, V | `ltp_hsv` |

1. Offline: separability margin for each (Phase 2 tooling), both datasets.
2. Online: all objects x 2 random rotations x 1 seed, both datasets (cheap
   once the models exist; the old "11 objects" referred to the retired set).

**Pre-registered rule** (write it in the doc before running): choose `ltp_hsv`
only if it beats `ltp_rgb` by >= 3 accuracy points on both datasets; otherwise
choose `ltp_rgb` (simpler, no hue wraparound handling). The losing variant goes
in the paper as an ablation row, not discarded.

## Phase 4 -- Calibration (0.5 day, offline; both lightings)

For every feature used in Phase 5 (color LTP winner, `ltp`, the Gabor
features), and separately for each lighting scenario's training graphs (a
deployed system calibrates on its own training data), choose tolerance by the
same rule on training graphs only:
maximize mean evidence(within) - mean evidence(across). Fix feature weight
across features (e.g. 1.0 each, normalized by Monty). Identical procedure for
LTP and Gabor is what makes the comparison fair.

## Phase 5 -- Main comparison, ambient then directional (2-3 days compute)

Gabor feature, implemented on days 1-2 because the main trainings store it:
bank of 4 orientations x 3 wavelengths (`skimage.filters.gabor_kernel`),
per-channel mean and std of response magnitude -> 24 values per channel,
L2-normalized, Euclidean distance. Feature keys `gabor` (grayscale),
`gabor_rgb` and `gabor_hsv` -- implement both color variants (one per-channel
function over different channel stacks) so the Phase 3 winner's Gabor match
exists without retraining. Implement as a peer of the LTP functions in
`sensor_processing.py` with unit tests, and an FFT path so the timing
comparison is against a reasonable Gabor, not a naive one.

**Conditions** (leading columns = which non-shape features are on):

| center-pixel `hsv` | texture feature | role |
|---|---|---|
| off | none | shape-only floor |
| on | none | **baseline** (Monty default) |
| off | `ltp` | texture without color |
| off | color LTP winner | color texture alone |
| on | color LTP winner | **proposed** |
| off | `gabor` | competitor, grayscale |
| off | `gabor_rgb` | competitor, color |
| on | `gabor_rgb` | competitor, matched to proposed |

**Design:** 8 conditions x 2 datasets x ~55 objects x 5 random rotations x 2
seeds = ~550 episodes per cell, ~8,800 per lighting scenario (raise
rotations/seeds if the Prelim C budget allows). Same seeds across conditions
and across lighting scenarios, so every test is paired by (object, rotation,
seed). Training: 4 combined-feature runs (2 datasets x 2 lightings; 14
predefined rotations; 50-60 objects x 14 = 700-840 episodes each).

**Lighting scenarios.**
- *Ambient (primary).* The full design above; the headline table.
- *Directional (second scenario).* Same conditions, objects, rotations and
  seeds, with directional-lit models and the directional Phase 4 tolerances.
  Its evals start as soon as the directional models finish and share the eval
  queue with ambient (ambient first). If the budget cannot afford all 8
  conditions, keep the planned-comparison conditions: baseline, proposed,
  `ltp`, Gabor matched. Also evaluate the directional models at the ambient
  tolerances, to separate "lighting changed the features" from "lighting
  changed the best tolerance".
- *Optional, if budget remains:* cross-lighting (train ambient, test
  directional) -- eval-only, a domain-shift result.

**Metrics:** `percent_correct`, `avg_convergence_steps` (+ SEM),
percent timed out (used most-likely hypothesis), and steps-to-correct curves
(fraction of episodes recognized by step k) -- the curve shows "faster" better
than a mean.

**Statistics:** exact McNemar (accuracy) and Wilcoxon signed-rank (steps),
paired by (object, rotation, seed); Holm correction over the planned
comparisons (proposed vs baseline, proposed vs `ltp`, proposed vs Gabor
matched), within each lighting scenario; bootstrap 95% CIs on differences.
Across scenarios, report the interaction -- (proposed - baseline) under
directional minus under ambient -- with a bootstrap 95% CI resampling
(object, rotation, seed) units.

**Pre-registered expectation** (record the outcome in the progress doc): the
proposed condition beats the baseline in both scenarios; the interaction CI
may exclude zero, but the sign of (proposed - baseline) does not flip under
directional light. A flip is reported as a limitation, not dropped.

## Phase 6 -- Wall-time (1 day, runs in parallel with Phase 5)

Robotics claim needs time, not steps. Three levels:

1. *Micro*: per-patch extraction time, 64x64, 10k real patches dumped from
   Phase 5, single thread (`taskset`, `OMP_NUM_THREADS=1`), median and p95:
   `ltp`, `ltp_rgb`, `ltp_hsv`, `gabor`, `gabor_rgb` (spatial and FFT).
2. *Per step*: total sensor-module + LM time per step for each condition
   (wrap `CameraSM.step` and LM `matching_step` with `time.perf_counter`).
3. *End to end*: **time to correct recognition** = steps x per-step time,
   per episode. This is the headline: fewer steps (accuracy benefit) times
   cheaper steps (compute benefit).

Report against a robotics budget (e.g. 33 ms/frame at 30 Hz) and add one
run on a low-power CPU (e.g. a Raspberry Pi / Jetson-class board) if available.

## Phase 7 -- Robustness (1.5 days compute, ambient; proposed vs baseline vs Gabor matched)

Lighting is not an ablation; it is a scenario of the main comparison (Phase 5).
Noise and footprint mismatch are eval-only (with `target_size` 0.12 m, the
mismatch test sets are 0.06 m -- a few minutes to build -- and the day-1 0.25 m
build); only the LTP threshold levels need
retraining, on 3DC-Texture only.

| Ablation | Levels | Why |
|---|---|---|
| Sensor noise | off, noise transform on | real cameras |
| Train/test footprint mismatch | train at chosen footprint, test at 0.5x and 2x | the distance/view concern directly; LTP is not scale-invariant, Gabor partly is |
| LTP threshold | 3, 5, 10 | sensitivity to the one LTP knob |

5 rotations x 2 seeds per cell is enough for ablations.

## Schedule (about 12 working days, down from ~20)

Three ways the schedule is shortened, in order of time saved:

1. **4 trainings instead of ~20**: one combined-feature pretraining per
   dataset x lighting (see "Train once per dataset x lighting").
2. **Parallel tracks** on the 64-core, 3-GPU machine (`run_parallel.py`):
   compute runs while code is written, and directional runs queue directly
   behind ambient instead of after the ambient results.
3. **Nothing waits on a cheap step**: all builds happen on day 1 at all three
   sizes; Phase 3 and Prelim B reuse the main models.

Critical path: builds -> Phase 2 subset trainings -> `target_size` decision
(+ Gabor done, Prelim C passed) -> 4 main trainings -> Phase 4 calibration ->
Phase 5 evals -> stats.

| Day | Compute track | Code / analysis track |
|---|---|---|
| 1 | Build both full sets + Phase 2 subsets at 0.12 / 0.25 / 0.5 m (~1 h); Phase 2 subset trainings (ambient, 2 rotations, parallel, ~15 min); Prelim A automated checks at all sizes, both lightings | Ambient + directional env configs for the new sets; combined-feature sensor module; location-specificity AUC; start Gabor (`gabor`, `gabor_rgb`, `gabor_hsv`) |
| 2 | Prelim C at the chosen size (smoke, determinism, combined-model equivalence, step caps, budget); Prelim A contact sheets | Phase 2 analysis -> `target_size` (can finish on day 1 once the AUC function exists); finish Gabor + unit tests + FFT path; per-step timing hooks (Phase 6) |
| 3 | Launch the 4 main trainings (ambient first on GPUs, directional alongside) | Eval configs for all conditions x lightings + snapshots; stats/plotting scripts on Prelim data |
| 4 | Prelim B (ambient models); Phase 3 evals | Phase 4 calibration (ambient, then directional as its models land); Phase 3 decision |
| 5-7 | Phase 5 evals: ambient, then directional (shared queue); Phase 7 noise + footprint-mismatch evals fill idle slots | Phase 6 micro timing in a quiet window (single pinned core, no concurrent load); start figures |
| 8 | Phase 7 threshold retrains + evals (3DC-Texture, ambient); optional cross-lighting | Stats: McNemar / Wilcoxon / Holm, interaction CIs |
| 9-12 | -- | Figures, writing |

Slippage risk sits on days 2-4: a failed Prelim B means editing recipes,
rebuilding (minutes) and retraining the affected dataset (hours), not days.

## Risks and fallbacks

- **Center-pixel `hsv` already near 100% on a dataset** -> caught by Prelim B;
  replace the color-outlier variants or the base composition and rebuild.
- **Some families have few close-colored materials** (e.g. the sofa's base
  fabric_23 is near-white, so its nearest fabrics still differ visibly in
  brightness) -> accept, or pick a different base composition for that class.
- **Dark materials are nearly invisible under the default lighting** -> caught
  by Prelim A's brightness check; add a brightness floor to variant selection.
- **The patch rarely visits the varied slot on some shapes** -> those objects
  test shape, not texture; replace the shape (Prelim A contact sheet).
- **50-60 object eval too slow** -> Prelim C budget decides rotations/seeds;
  never cut objects (dataset breadth is the point of this revision).
- **No footprint gives location-specificity** -> use multi-patch or smaller
  zoom-in sensor; report as a limitation of single-patch color features.
- **Gabor beats LTP on accuracy** -> the paper becomes an accuracy-vs-cost
  trade-off; the time-to-correct-recognition metric still decides it.
- **Compute overrun** -> drop to 2 seeds in Phase 5 before dropping a dataset;
  under directional light, cut to the four planned-comparison conditions
  before cutting a dataset or the scenario.
- **Combined-feature training not equivalent to per-feature training**
  (caught in Prelim C) -> per-feature trainings; the schedule grows by ~3 days.
