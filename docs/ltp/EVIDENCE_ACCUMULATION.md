# How Evidence Matching Works (and what `tolerance` / `weight` mean)

A walkthrough of how Monty's `EvidenceGraphLM` turns sensor observations into an
object recognition, and exactly where the `tolerances` and `feature_weights`
parameters enter. Code references point at the matching pipeline.

Key files:
- `src/tbp/monty/frameworks/models/evidence_matching/feature_evidence/calculator.py`
- `src/tbp/monty/frameworks/models/evidence_matching/feature_evidence/scorer.py`
- `src/tbp/monty/frameworks/models/evidence_matching/hypotheses_displacer.py`

---

## The big picture: what is being "matched"

Monty doesn't compare whole objects at once. It maintains thousands of
**hypotheses**, each one a specific guess of the form:

> "I'm currently touching **object X**, at **location L** on that object, with the
> object held in **orientation R**."

Every sensor observation (a "step") nudges each hypothesis's **evidence** score up
or down based on how well the *observed* sensation matches what *that hypothesis
predicts* you should be sensing. Recognition happens when one object's best
hypothesis pulls far enough ahead of the rest.

`tolerance` and `weight` are the two knobs that control how a single observation
turns into an evidence change.

---

## Step 1 — Per-feature: how close is "matching"? (this is `tolerance`)

A stored graph node holds feature values (an `hsv`, a curvature, an `ltp`
histogram, ...). The sensor gives observed values. For each feature, Monty
computes a **difference**:

- numeric (e.g. curvature): `|observed - stored|`
- circular (hue): wrapped angular distance
- histogram (`ltp`): a distance metric (chi-square or Hellinger)

Then **tolerance** converts that raw difference into a per-feature score in
`[0, 1]` (`calculator.py`, near the end of `calculate`):

```
feature_evidence = clip(tolerance - difference, 0, inf) / tolerance
```

So tolerance is the **width of the "still counts as a match" window** for that
feature:

| difference vs tolerance        | score             |
|--------------------------------|-------------------|
| difference = 0 (identical)     | **1.0** (perfect) |
| difference = 1/2 x tolerance   | 0.5               |
| difference >= tolerance        | **0.0** (mismatch)|

A *small* tolerance = strict ("only near-identical values count"). A *large*
tolerance = permissive ("even pretty different values still score high").

This is why a baseline `ltp` tolerance of 20 was broken: every node's chi-square
distance was below 20, so every node scored ~1.0 -- the feature could never say
"no."

---

## Step 2 — Across features: how much does each one matter? (this is `weight`)

A node has many features, each now with its own `[0,1]` score. **Weight** combines
them into a *single* feature-evidence number via a **weighted average**
(`calculator.py`, final line of `calculate`):

```
feature_evidence = sum(weight_i * score_i) / sum(weight_i)      # still in [0, 1]
```

Weight is **relative importance**. With `hsv ~ 1`, `curvature ~ 1`, `ltp = 20`, the
average is dominated by the `ltp` score -- texture effectively *decides* the
feature evidence, and the others just nudge it. With `ltp = 1` (the old default),
texture was one equal voice among several, so a great texture match couldn't move
the needle much.

> Key intuition: **tolerance decides whether a feature matches; weight decides how
> much that match counts** toward the node's overall feature score.

The combination happens in `DefaultFeatureEvidenceScorer.__call__`
(`scorer.py`), which calls the calculator and multiplies the result by
`feature_evidence_increment` (a global scale on the whole feature contribution
relative to pose).

---

## Step 3 — Features are only half of it: geometry / pose evidence

Matching texture/color isn't enough -- the *geometry* has to line up too.
Separately from the feature evidence, Monty computes **pose evidence**
(`hypotheses_displacer.py`, `_get_pose_evidence_matrix`): it compares the observed
surface normal and curvature direction against the stored node's, scoring the
angular agreement into roughly `[-1, +1]` (aligned -> positive, opposed ->
negative).

Note pose has its *own* weights too: `feature_weights["pose_vectors"]` scales the
surface-normal and curvature-direction terms directly (it does **not** go through
the `[0,1]` weighted average -- pose is handled in a separate additive channel).

The per-node, per-step evidence is then **added together**
(`hypotheses_displacer.py`, `radius_evidence = radius_evidence + hypothesis_radius_feature_evidence`):

```
node_evidence = pose_evidence(~[-1,1])  +  feature_evidence([0,1] * feature_evidence_increment)
```

There's also a **location gate**: only nodes within `max_match_distance` of where
the hypothesis predicts you are get considered at all (others are forced to -1).
Among the nearest neighbors that pass, Monty takes the **max**
(`np.max(..., axis=1)`) -- the best-matching nearby node represents the hypothesis
this step.

So the geometry (location + pose) answers *"is there a point here, oriented like
this?"* and the features answer *"and does that point look/feel like what I'm
sensing?"* Both must agree for a high score.

---

## Step 4 — Accumulating over steps

Each hypothesis carries its evidence forward (`hypotheses_displacer.py`):

```
new_evidence = old_evidence * past_weight  +  this_step_evidence * present_weight
```

As you move across the object, the *correct* object/pose hypothesis keeps getting
points added at every step (everything keeps matching), while wrong hypotheses
stall or lose points as soon as features or geometry stop agreeing. Evidence is an
accumulating vote, not a single comparison.

---

## Step 5 — Declaring a match

Recognition is declared when the **most-likely hypothesis's object** stands out
from the others by more than `x_percent_threshold` (a confidence margin).

On the reskinned balls, the geometry of all 4 baseballs is identical, so pose
evidence is the *same* for all of them -- the *only* thing that can create that
separating margin is the texture (`ltp`) feature. That's precisely why the
weight/tolerance on `ltp` is the whole game for this experiment.

---

## Tying it back to the tuning

| Parameter            | Controls                                | Symptom if wrong                                                                                   |
|----------------------|-----------------------------------------|----------------------------------------------------------------------------------------------------|
| **tolerance** (`ltp`)| width of the texture match window       | too large (20) -> every texture "matches" -> no discrimination. too small -> correct textures score 0 -> no evidence at all |
| **weight** (`ltp`)   | texture's share of the feature evidence | too small (1) -> texture can't influence the outcome; correct/wrong balls tie                      |

Measured chi-square distances for the stored 72-bin `ltp` histograms (within-texture
median 0.20 vs wrong-texture median 0.95; see `LTP_BRANCH_COMPARISON_ANALYSIS.md`
section 4b) are exactly what let us set tolerance ~ 0.5: wide enough that a true
match (0.20 -> score 0.6) survives, narrow enough that a wrong texture
(0.95 -> score 0) is rejected -- and weight 20 makes that score actually decide the
match.
