# LTP Texture Experiments: Experimental Protocol

The protocol for the Local Ternary Pattern (LTP) texture-feature experiments on the
`bica-exp` branch: what is being tested, how the object models are trained, how
recognition is evaluated, how LTP enters the matching decision, and which conditions
are swept.

Companion documents: `EVIDENCE_ACCUMULATION.md` (how tolerance/weight enter the
evidence computation), `LTP_BRANCH_COMPARISON_ANALYSIS.md` (why the original
chi-square configuration failed and how the tolerance was calibrated),
`3DCOMPAT_EXPERIMENTS.md` (the 3DCoMPaT++ dataset pipeline).

---

## 1. Objective

Test whether a rotation-invariant LTP texture feature lets Monty discriminate objects
that are **geometrically identical and differ only in surface texture** — the case
where pose and curvature evidence are provably uninformative, so any discrimination
must come from texture.

## 2. Object sets

**Primary — retextured YCB** (`$MONTY_DATA/retextured_ycb`), 11 objects:

| Group | Objects | Role |
|---|---|---|
| Reskinned baseballs | `baseball`, `porous_baseball`, `matted_baseball`, `flecked_baseball` | Identical geometry; texture is the only discriminator |
| Reskinned softballs | `softball`, `blotchy_softball`, `woven_softball`, `zigzagged_softball` | Same |
| Distinct controls | `mug`, `bowl`, `banana` | Verify the rest of the pipeline still works |

**Secondary — 3DCoMPaT++**: vase and jug in four material compositions each, plus
planter, stool, basket. Same shape-constant / material-varying design on a different
dataset. See `3DCOMPAT_EXPERIMENTS.md`.

## 3. Sensing and the LTP feature

Habitat **surface agent** (a distant-agent variant exists for the same objects), with a
64×64 patch sensor at zoom 10 and a 64×64 view-finder.

`CameraSM` emits, per observation: pose vectors (surface normal + curvature
directions), HSV, principal curvatures (raw and log), and the `ltp` histogram.

LTP extraction (`utils/sensor_processing.py`), following the split formulation of
Tan & Triggs:

- positive codes: neighbor ≥ center + 5; negative codes: neighbor ≤ center − 5
  (8 neighbors, radius 1, on the 0–255 grayscale patch)
- each code map is passed through **ROR rotation-invariant encoding**
- the two histograms are concatenated and L1-normalized → **72 bins**
- only **on-object pixels** contribute to the histogram

The sensor module additionally reports `ltp_pixel_stats` (patch mean intensity and
variance). This is metadata about the observation, not a matched feature; the matching
step uses it to decide whether to trust the texture signal (§5).

---

## 4. Training — building the object models

Training is **not** learning a classifier. It builds a graph model of each object, and
it is fully supervised: the object's identity and ground-truth pose are handed to the
model, so no matching or recognition happens at all.
`MontySupervisedObjectPretrainingExperiment.run_episode` calls
`switch_to_exploratory_step()` up front and skips the matching machinery entirely. The
episode is pure data collection.

**Episode = one object in one rotation.** Habitat places the object; a `GetGoodView`
positioning procedure on the view-finder walks the agent in until the object is at a
usable distance. The surface agent then crawls over the object under
`SurfacePolicyCurvatureInformed` with `use_goal_driven_actions: false` (the `goal0`
policy — during training there is no hypothesis to test, so the curvature-following
policy runs unguided). It holds a 0.025 m standoff, follows principal-curvature
directions with a bias that decays after 32 steps, and takes at least 8 general /
12 heading steps before changing course.

**Each step** yields a feature dict at a 3D location (pose vectors, HSV, curvatures,
72-bin LTP histogram) which goes into the LM's buffer.

**Step budget.** `num_exploratory_steps: 1000` is the *length* of the exploration
phase, not a ceiling on it: `min_steps` returns it for the exploratory step type, and
the episode ends as soon as `exploratory_steps > 1000`. Only steps the sensor and LMs
actually process increment that counter — a **motor-only step** (e.g. the patch has
come off the object and the agent must get back on it) bumps `episode_steps` and
`total_steps` but not `exploratory_steps`. So an episode always yields 1000 processed
observations, but may take more than 1000 simulator steps to get them.
`max_total_steps` is the safety net for exactly that gap: the loop breaks on
`model.episode_steps >= max_total_steps`, and `episode_steps` counts everything,
motor-only steps included. Its job is to stop an episode where the agent is thrashing
on a thin or awkward object and never accumulating its 1000 useful observations; in
the normal case it never fires.

> Note on the two `max_total_steps`. `monty_args.max_total_steps: 2500` (from the
> `graph_exp1000_e3_t3_tot2500` group) is stored on the Monty model but is **not** what
> the loop reads. The experiment reads `config["max_total_steps"]`, set directly in the
> experiment yaml — **6000** in the LTP configs. That is the value in force; the 2500 is
> inert. `max_train_steps: 1000` bounds the *matching* phase and is therefore also dead
> weight in pretraining, which never enters matching. (`MontyExperiment` auto-raises
> `max_total_steps` if it is ever configured below `num_exploratory_steps`, so the two
> cannot contradict each other.)

**Turning the buffer into a graph.** At episode end the ground-truth label, position
and rotation are written into the LM (`detected_object`, `detected_rotation_r` from the
target quaternion), so the collected points are placed into the object's model frame
exactly, with no pose-estimation error. `post_episode` then hands the buffer to
`GraphMemory._build_graph` → `GraphObjectModel.build_model`, where
`graph_delta_thresholds` decide whether an observation is distinct enough from what is
already stored to become a new node. In `displacement_delta_d0001_ltp.yaml` a point is
kept if it differs by more than:

- 1 mm in space, **or**
- π/8 in surface normal, **or**
- 1 in either log curvature, **or**
- 0.1 in hue.

Everything else is a redundant re-observation and is dropped. This is why LTP must be
*sensed* during training even though it is not *matched*: every stored node carries the
LTP histogram observed there, and that is what evaluation later compares against.

**Repeats.** Each object is visited once per epoch for **14 epochs** — one per rotation
in `rotations_all`, the 14 fixed viewpoints that jointly give good coverage. Successive
epochs *extend* the same graph (`_extend_graph`), so all 14 views accumulate into one
model per object. Seed 42, `min_train_steps: 3`.

**Output.** 11 graphs in
`$MONTY_MODELS/my_trained_models/surf_pre_training_11retextured_obj_ltp/pretrained/model.pt`,
each a point cloud of nodes carrying (location, pose vectors, HSV, curvatures, LTP
histogram).

**Which conditions need their own training run.** Anything that changes *what gets
stored in the nodes* — LTP encoding (ROR / uniform / multi-scale), grayscale vs
per-channel `ltp_rgb`, on-object masking, ambient vs default lighting — requires
re-running pretraining, because the stored histograms themselves change. Anything that
only changes *how stored histograms are compared* — distance metric, tolerance, feature
weight — is an eval-time knob and reuses a single pretrained model. This is why the
metric/tolerance/weight sweep is cheap and the encoding/lighting sweep is not.

---

## 5. Evaluation — object recognition

The pretrained graphs are loaded and frozen. Each episode presents one object in a
**random rotation** (`RandomRotation` sampler, so essentially never one of the 14
training rotations), and Monty must recognize both *which object* it is and *what pose
it is in*, from scratch, by moving over it. **10 epochs × 11 objects = 110 episodes**,
seed 42.

**Hypothesis space.** `MontyForEvidenceGraphMatching` with `EvidenceGraphLM` maintains,
for every known object, a large set of hypotheses of the form *"I am on object X, at
node location L, with the object in rotation R."* The space is seeded at the first
observation from graph nodes consistent with the sensed pose; thereafter each
hypothesis is displaced by the agent's actual movement and rescored.

**Scoring a step.** For each hypothesis, `BurstSamplingHypothesesUpdater`
(`max_nneighbors: 5`) looks at the up-to-5 stored nodes nearest to where that
hypothesis predicts the sensor now is; nodes farther than `max_match_distance: 0.01`
(1 cm) are excluded outright. For each candidate node, two quantities are computed and
**added**:

1. **Pose evidence**, roughly in [−1, +1], from the angular agreement between the
   observed surface normal / curvature direction and the stored node's (scaled by the
   `pose_vectors` weight).
2. **Feature evidence**, in [0, 1]: the weighted average over HSV, log curvatures and
   `ltp`, where each feature contributes
   `clip(tolerance − distance, 0, ∞) / tolerance`.

The **best** of the ≤5 neighbouring nodes represents that hypothesis for the step, and
the hypothesis's running evidence is updated as a weighted blend of its old evidence and
this step's score. With `evidence_threshold_config: all`, every hypothesis is updated
each step rather than only the top ones. Evidence is therefore an accumulating vote
across the trajectory: the correct object/pose keeps collecting positive increments,
while wrong ones stall as soon as geometry or texture stops agreeing. (Full derivation
in `EVIDENCE_ACCUMULATION.md`.)

**The LTP term.** Its distance is the **Hellinger** distance between the two 72-bin
histograms — bounded in [0, 1], which replaced the original unbounded, asymmetric
chi-square. Current operating point:

| Feature | Weight | Tolerance |
|---|---|---|
| `ltp` | 2.0 | 0.50 |
| `hsv` | [2, 0.5, 0.5] | [0.1, 0.2, 0.2] |
| `pose_vectors` | 1, 1, 1 | — |
| `principal_curvatures_log` | 1, 1 | 1, 1 |

A **reliability gate** forces the LTP weight to 0 *for that observation* when the patch
is dark (mean intensity < 60), saturated (mean > 230), or nearly uniform (variance
< 400) — regimes where the texture signal is dominated by sensor noise.

**Why LTP is the whole game here.** For the eight reskinned balls, pose evidence is by
construction *identical* across the four members of a group: the geometry is the same
mesh. The only term that can separate them is the LTP histogram distance. This is why
weight 0 (the `w0` config) is the meaningful control — it reduces the model to
geometry-only and should collapse the ball groups.

**Deciding a match.** `_threshold_possible_matches` sums evidence per object and keeps
every object within `x_percent_threshold` (20%) of the maximum:
`th = max_evidence − 0.20 × max_evidence`. When exactly one object survives that margin
(and the pose has converged), the LM enters terminal state `match`; the episode ends
once `min_lms_match: 1` LM is in that state, but never before `min_eval_steps: 20`. If
no object clears the bar, the state is `no_match`. If neither happens within
`max_eval_steps: 500` matching steps (or 6000 total steps), the episode is a `time_out`,
and the most-likely hypothesis at that moment is still recorded.

**Active sensing.** Unlike training, evaluation uses the `goal1` policy
(`use_goal_driven_actions: true`) with the `EvidenceGoalGenerator`. Once the LM is
partway to a decision it proposes *where to move next* to maximally disambiguate the
remaining candidates (`goal_tolerances.location: 0.015`, `x_percent_scale_factor: 0.75`
so hypothesis-testing starts before the full classification margin is reached). The
trajectory is not a passive scan — the model steers the sensor toward locations it
expects to be discriminative.

**The key asymmetry to keep in mind:** training sees 14 fixed rotations *and is given
the labels*; evaluation sees unseen random rotations and gets nothing. That gap is what
makes the rotation-invariant (ROR) encoding load-bearing — a rotation-variant `uniform`
encoding stores histograms that are only valid at the training viewpoints, which is
exactly what the ROR-vs-uniform ablation tests.

---

## 6. Tolerance calibration

The tolerance is not guessed. `MontyLTPDistanceExperiment` (`plot_ltp_distances*`)
loads a pretrained model and measures the **within-object** vs **across-object**
histogram-distance distributions under each candidate metric (chi-square, Hellinger,
plus L1/L2/intersection/Bhattacharyya/correlation/KL/Jensen-Shannon for comparison; the
chi-square and Hellinger implementations are the *actual* Monty matching functions, so
the plotted distances are exactly what Monty computes).

Because feature evidence is non-negative and additive, a mismatching node is never
*penalized* — all of LTP's discriminative work is done by the **gap** between the
evidence a correct node accrues and what a wrong one accrues. The tolerance is therefore
chosen to maximize that gap (`best_tolerance`), not by a hard separating threshold
(`best_threshold`, Youden's J, is retained only as a reference; it picks systematically
too-tight values that clip correct-object nodes to zero).

---

## 7. Conditions swept

| Axis | Values | Retrain needed? |
|---|---|---|
| Distance metric | chi-square (original) vs Hellinger | No |
| Tolerance | chi-square {0.1, 0.3, 0.5, 1, 2}; Hellinger {0.03 … 0.50} | No |
| LTP weight | {0, 1, 2, 5, 20} — **weight 0 = no-texture control** | No |
| Encoding | ROR (rotation-invariant) vs `uniform` (rotation-variant); multi-scale radii 1/2/3 | **Yes** |
| Color | grayscale `ltp` (72 bins) vs per-channel `ltp_rgb` (3 × 72 bins) | **Yes** |
| On-object masking | masked vs unmasked histograms | **Yes** |
| Illumination | default Habitat lighting vs `AMBIENT_LIGHTS` | **Yes** |
| Agent | surface vs distant | **Yes** |
| Dataset | retextured YCB vs 3DCoMPaT++ | **Yes** |

---

## 8. Measures

Each episode writes a row to `eval_stats.csv` with `primary_performance` ∈ {`correct`,
`correct_mlh`, `confused`, `confused_mlh`, `no_match`, `time_out`, `patch_off_object`},
plus `num_steps` and `rotation_error`. Two accuracies are reported:

- **strict** — `primary_performance == "correct"`: converged on the right object within
  the 20% margin.
- **inclusive** — also counts `correct_mlh`: the right object was leading at timeout but
  never opened a 20% gap.

The difference between the two is itself informative: a texture feature that is
discriminative but *weak* shows up as `correct_mlh` rather than `correct`. Also reported:
mean steps to convergence and mean rotation error. `scripts/compare_eval_runs.py`
tabulates these across runs; everything is mirrored to wandb.

The headline contrast is accuracy on the **eight same-geometry reskins** versus the
**three distinct controls**, since only the former isolates the texture feature.

---

## 9. Running it

```bash
conda activate tbp.monty
cd ~/Projects/tbp.monty

# Train (once per sensing condition)
python run.py experiment=surf_pre_training_11retextured_obj_ltp

# Evaluate (reuses the pretrained model across matching-knob conditions)
python run.py experiment=randrot_11retexturedobj_surf_agent_ltp

# Calibrate the tolerance against the stored histograms
python run.py experiment=plot_ltp_distances_surf

# Compare runs
python scripts/compare_eval_runs.py <results_dir> [<results_dir> ...]
```

Every yaml in `conf/experiment/` is snapshot-tested. After editing an experiment (or
anything it composes), regenerate with
`python src/tbp/monty/conf/update_snapshots.py` and commit the diff.

---

## 10. Known inconsistencies

- `randrot_11retexturedobj_surf_agent_ltp.yaml` has `run_name: ..._t050` while the LM
  config sets tolerance 0.50 and weight 2.0. Currently consistent, but the name will not
  track a retune.
- `amb_randrot_11retexturedobj_surf_agent_ltp.yaml` has `run_name: ..._t018` while
  composing the same tolerance-0.50 LM config — the filename encodes a tolerance it is
  not actually using.
