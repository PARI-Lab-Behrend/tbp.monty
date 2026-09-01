# BICA 2026 — Submission 96 Revision Checklist

**Paper:** Texture Feature Enhanced Sensory Module for Improved Object Recognition in a Neocortex-inspired Embodied Sensorimotor Agent (Monty)

## Score summary

| Criterion | R1 | R2 | R3 |
|---|---|---|---|
| Scope and Relevance | 5 | 4 | 3 |
| Novelty and Significance | 5 | 4 | 4 |
| Rigor and Soundness | 4 | 4 | 4 |
| Clarity, Style, Completeness | 4 | 3 | 4 |
| **Overall** | **+1 weak accept** | **0 borderline** | **+1 weak accept** |
| CSR nomination | no | no | no |

**Read of the situation:** novelty is not in question (5/4/4). The losses are in rigor and presentation, which are the cheapest things to fix. R1's text says "major revisions" but the score is +1 with an explicit accept-if-addressed — treat the tone as louder than the verdict. Highest-cost requests (real robot, SOTA benchmarking) should mostly be argued down rather than attempted.

Each checklist item below is annotated with `↪ *Rn:*` and the exact review sentence it maps to (see [Review](#review)). Items with no annotation do not correspond to anything a reviewer actually wrote.

---

## Tier 1 — Fix immediately (hours, not days)

Pure credibility leaks. This list is most of what drove R2's Clarity 3.

- [x] Remove template placeholders: `Authors Suppressed Due to Excessive Length` and `Title Suppressed Due to Excessive Length` in running heads — set `\titlerunning{}` and `\authorrunning{}` in the llncs preamble
  - ↪ *R2:* "The headers still include template text such as "Authors Suppressed Due to Excessive Length" and "Title Suppressed Due to Excessive Length,""
- [x] Fix duplicated bullet on the `uniform` entry in §3.3 (renders as `– –`)
  - ↪ *R2:* "a duplicated bullet point appears in the list of encoding schemes."
- [x] Standardize spelling to **Bhattacharyya** throughout — §2.5 currently has "Bhattacharya"; abstract and §3.4 have "Bhattacharyya"; reference [1] settles it
  - ↪ *R2:* "The spellings "Bhattacharya" and "Bhattacharyya" are used inconsistently"
- [x] Rename the dataset — `11retextured` is being read as a typo. Use `YCB-11R` or `11-Retextured`, define once, apply everywhere (abstract, §3.2, §4, figures, tables)
- [x] Rename `avg_convergence_rate` → `avg_steps_to_convergence` (it is a count, not a rate). Update §3.5, Tables 1–3, and Discussion
  - ↪ *R2:* ""avg_convergence_rate" actually refers to the average number of steps to convergence, so the metric should be renamed"
- [x] Add missing SEM to Table 3, HSV-only row (currently `53.12` with no ±)
  - ↪ *R2:* "the missing error information in Table 3 should be added."
- [x] Standardize the `± SEM` annotation across Tables 1–3 — only Table 1's header currently mentions it
  - ↪ *R2:* "it reports only means and partial standard errors."
- [ ] Clarify "split (Completed) LTP" terminology — CLTP in Rassem & Khoo [26] means sign + magnitude + center. If only the sign-split is used, say so and define "split LTP" as your own term
- [ ] Language pass for run-on and ungrammatical sentences (this is R3's entire substantive comment)
  - ↪ *R3:* "There are small imperfections in the paper text. The text of the paper should be check attentively."
  - ↪ *R2:* "some sentences are overly long or grammatically incorrect."

---

## + Tier 2 — Statistical rigor (highest-leverage block)

All three reviewers scored Rigor 4. R2 supplied the recipe.

- [x] Report full experimental protocol in §3.4: number of random rotations per object, how rotations were sampled, whether the same rotation set was reused across feature conditions, and random seeds — the reader currently cannot reconstruct 220 vs. 110 vs. 231
  - ↪ *R2:* "The paper does not clearly report the number of repetitions for the random-rotation experiments, the random seeds, or any statistical significance tests."
- [x] **Do the arithmetic before committing to any claim.** Ambient: 93.18% → 96.81% at n=220 is ~15 errors → ~7. Directional: n=110, ~12 → ~6. Convergence SEMs of ±15 on a 35-step gap is under 2σ unpaired. None of these is comfortably significant as reported
  - ↪ *R2:* "Although the manuscript uses expressions such as "significantly increases," it reports only means and partial standard errors."
- [x] Run significance tests:
  - ↪ *R2:* "...the random seeds, or any statistical significance tests."
  - [ ] If rotations were paired across conditions → **McNemar** on accuracy, **paired Wilcoxon** on per-episode step counts (pairing buys substantial power on convergence)
  - [ ] If not paired → re-run paired. This is a seed change, not a new experiment
- [x] Soften "significantly increases" in the Discussion unless a test backs it. Stronger honest framing: *consistent directional improvement across four independent conditions* + a clearly significant convergence effect
  - ↪ *R2:* "Although the manuscript uses expressions such as "significantly increases," it reports only means and partial standard errors."
- [x] Neutralize the tuning-bias charge (R2 believes metric + tolerance were selected on the evaluation data):
  - ↪ *R2:* "The LTP distance metric and tolerance parameters also appear to have been selected using the same dataset as the final evaluation, which may introduce tuning bias. The authors should separate the parameter-selection set from the final test set."
  - [ ] Fig. 5 is labeled "pretrained" — if training-rotation patches were used, **state this explicitly in §3.4**; right now it is unverifiable
  - [ ] If not, redo threshold selection on a held-out split (YCB patches or training rotations only) and report the test number under the frozen threshold
- [ ] Add tolerance sensitivity sweep (~0.3–0.7) — small curve or 4-row table
  - ↪ *R1 (#4):* "the paper does not conduct parameter sensitivity analysis on the Hellinger distance tolerance threshold and multi-scale patch radius"
- [ ] Add multiscale patch-radius ablation (answers R1 #4 and R2's tuning point with the same experiment)
  - ↪ *R1 (#4):* "the paper does not conduct parameter sensitivity analysis on the Hellinger distance tolerance threshold and multi-scale patch radius"

---

## Tier 3 — Added experiments, ranked by cost/benefit

- [ ] **Compare against LBP — do this one.** R2's strongest legitimate criticism is that LTP is never justified against obvious alternatives. Nearly free given the existing pipeline, since split LTP is structurally two LBP-style codes
  - ↪ *R2:* "The main baseline is Monty's original HSV feature, with no comparison against LBP, CLBP, Gabor, HOG, or other lightweight texture descriptors. Therefore, the advantages of choosing LTP are not yet fully demonstrated."
  - ↪ *R1 (#2):* "without benchmarking against mainstream embodied vision agents, HTM-based recognition models or classic lightweight texture recognition architectures."
- [x] Add Gabor if time allows
  - ↪ *R2:* "no comparison against LBP, CLBP, Gabor, HOG, or other lightweight texture descriptors."
- [x] **Skip HOG** — gradient/shape descriptor, arguably redundant with Monty's morphological features. Say so in one sentence instead of running it
  - ↪ *R2:* "no comparison against LBP, CLBP, Gabor, HOG, or other lightweight texture descriptors."
- [x] Complete the ablation grid (cheap config toggles, answers R1 #4):
  - ↪ *R1 (#4):* "it lacks independent control groups that remove morphological features or test single texture feature performance."
  - [ ] LTP-only, no HSV, on 11retextured
  - [ ] Neither feature on 11retextured — this row exists for YCB (Table 3) but is missing for the harder dataset
- [ ] Dataset diversity (partial fix): 8 of 11 objects are retextured spheres. Add retextured mug/bowl/banana variants for non-spherical texture cases — modest Blender effort, most defensible widening without new data collection
  - ↪ *R2:* "The custom dataset contains only 11 objects, most of which are manually retextured versions of a small number of models, so its scale and diversity are limited. The authors are encouraged to include more materials, shapes, and background conditions..."

---

## Tier 4 — Framing (cheap, targets the two weakest scores)

- [ ] Deepen biological grounding to ~½ page (R1 #1). Currently one sentence linking split LTP to center-surround RFs via Kuffler [15]. Expand to cover:
  - ↪ *R1 (#1):* "it lacks in-depth linkage with cortical column, HTM and core Thousand Brains Theory mechanisms of Monty."
  - [ ] Multiscale radii ↔ RF sizes and spatial-frequency channels in V1
    - ↪ *R1 (#1):* "The paper does not elaborate how texture feature extraction simulates biological visual cortex multi-scale texture perception"
  - [ ] Rotation-invariance choice ↔ orientation selectivity
    - ↪ *R1 (#1):* "The paper does not elaborate how texture feature extraction simulates biological visual cortex multi-scale texture perception"
  - [ ] Where a non-morphological texture feature sits in the CMP, and how that maps to a cortical column's feature-at-pose representation
    - ↪ *R1 (#1):* "it lacks in-depth linkage with cortical column, HTM and core Thousand Brains Theory mechanisms of Monty"
- [x] Add one explicit paragraph in the intro tying the sensory module to biologically inspired cognitive architecture concerns (R3 scored Scope 3, called the theme "indirectly related")
  - ↪ *R3:* "This theme is indirectly related to topics of the conference. However, the theme could be interesting to some participants of the conference, so the paper can be accepted."

---

## Push back — do not attempt

- [ ] **Real robot validation (R1 #3)** — out of scope for a conference paper, already stated as future work in the Conclusion. In the response letter, cite the sim-to-hardware plan and the Habitat directional-lighting condition as a partial stress test
  - ↪ *R1 (#3):* "All experiments are limited to the Habitat simulation environment, without deploying the improved sensory module on physical robotic arms for real-world object recognition tests."
- [ ] **SOTA / HTM benchmarking (R1 #2)** — justification already exists in the Conclusion but is buried and reads as an afterthought. **Move it into the Discussion and expand it**, then point R1 there. The defensible concession is the LBP comparison above, which is what the criticism reduces to
  - ↪ *R1 (#2):* "The work only compares multiple LTP variants within the Monty framework, without benchmarking against mainstream embodied vision agents, HTM-based recognition models or classic lightweight texture recognition architectures."
- [ ] **"Remove morphological features" ablation (R1 #4)** — Monty's graphs *are* the morphological representation; removing them leaves no system. One sentence, not an experiment
  - ↪ *R1 (#4):* "it lacks independent control groups that remove morphological features or test single texture feature performance."

---

## Suggested order of work

1. **Tier 1** — one day, and it changes the reviewers' impression before they read anything else
2. **Significance testing** (Tier 2, arithmetic + tests) — do early, since the outcome may force rewording of the Discussion
3. **Tuning-bias fix, sensitivity sweep, LBP comparison**
4. **Tier 4 framing**
5. **Remaining ablations and dataset expansion** if the revision window allows

---

## Response letter notes

- Structure as point-by-point, reviewer by reviewer
- Roughly half the strategy here is *declining* requests — each decline needs a stated reason plus a concrete partial concession, never a flat refusal
- Lead each reviewer's section with what was changed, not with what was disputed

## Review

----------------------- REVIEW 1 ---------------------

SUBMISSION: 96
TITLE: Texture Feature Enhanced Sensory Module for Improved Object Recognition in a Neocortex-inspired Embodied Sensorimotor Agent (Monty)

----------- Scope and Relevance -----------
SCORE: 5 (Yes)
----------- Novelty and Significance -----------
SCORE: 5 (Yes)
----------- Rigor and Soundness -----------
SCORE: 4 (Can be improved)
----------- Clarity, Style, Completeness, English -----------
SCORE: 4 (Can be improved)
----------- Overall evaluation -----------
SCORE: 1 (weak accept)
----- TEXT:
This paper integrates split Local Ternary Pattern (LTP) texture descriptors into the neocortex-inspired Monty sensorimotor agent to address the limitation that the original system cannot distinguish same-shaped objects with different textures. Sufficient comparative experiments are conducted on YCB and self-built retextured datasets under two lighting conditions, verifying the recognition accuracy and convergence speed gains brought by multi LTP encoding schemes. The research closely matches BICA’s embodied cognitive agent research theme. Nevertheless, the manuscript has obvious defects in biological plausibility argumentation, comparative analysis with mainstream cognitive/vision models, real robot verification and ablation completeness, which require major revisions.

1. Insufficient Biological Inspiration Interpretation
Although the paper briefly mentions that split LTP conforms to center-surround receptive field characteristics, it lacks in-depth linkage with cortical column, HTM and core Thousand Brains Theory mechanisms of Monty. The paper does not elaborate how texture feature extraction simulates biological visual cortex multi-scale texture perception, weakening the biological interpretability required by BICA.

2. Lack of Comparative Experiments with State-of-the-Art Models
The work only compares multiple LTP variants within the Monty framework, without benchmarking against mainstream embodied vision agents, HTM-based recognition models or classic lightweight texture recognition architectures. It fails to quantitatively demonstrate the advantages of the proposed texture-enhanced sensory module against existing bio-inspired visual systems.

3. No Real Physical Robot Validation
All experiments are limited to the Habitat simulation environment, without deploying the improved sensory module on physical robotic arms for real-world object recognition tests. The generalization of the method in actual noisy, uneven illumination physical scenes cannot be proven, reducing the practical value for embodied intelligence research.

4. Incomplete Ablation and Control Experiments
The ablation test only combines HSV with different LTP modes; it lacks independent control groups that remove morphological features or test single texture feature performance. Besides, the paper does not conduct parameter sensitivity analysis on the Hellinger distance tolerance threshold and multi-scale patch radius, making the optimal feature configuration conclusion insufficiently robust.

This study has important academic value and innovation, but the current problems are relatively serious, requiring comprehensive major revisions. The authors need to deepen the analysis linking LTP texture extraction to biological visual cortex perception mechanisms, add quantitative comparisons with advanced bio-inspired vision agents, supplement physical robot real-scene verification experiments, and enrich complete ablation and parameter sensitivity control tests. The revised manuscript will be re-reviewed, and if all issues are properly addressed, it is recommended to be accepted as a conference paper for BICA 2026.
----------- CSR -----------
SELECTION: no


----------------------- REVIEW 2 ---------------------

SUBMISSION: 96
TITLE: Texture Feature Enhanced Sensory Module for Improved Object Recognition in a Neocortex-inspired Embodied Sensorimotor Agent (Monty)

----------- Scope and Relevance -----------
SCORE: 4 (Can be improved)
----------- Novelty and Significance -----------
SCORE: 4 (Can be improved)
----------- Rigor and Soundness -----------
SCORE: 4 (Can be improved)
----------- Clarity, Style, Completeness, English -----------
SCORE: 3 (Must be improved)
----------- Overall evaluation -----------
SCORE: 0 (borderline paper)
----- TEXT:
This paper adds a texture feature based on split Local Ternary Patterns (LTP) to Monty’s sensory module in order to distinguish objects with similar shapes but different surface textures. It compares three encoding schemes—ror, uniform, and multiscale—and evaluates them on the YCB dataset and a custom 11retextured dataset. The results show that LTP can improve recognition accuracy and reduce the number of steps required for convergence under some settings. Since the method requires no additional pre-training, it has some practical value.

However, the experimental comparisons are still limited. The main baseline is Monty’s original HSV feature, with no comparison against LBP, CLBP, Gabor, HOG, or other lightweight texture descriptors. Therefore, the advantages of choosing LTP are not yet fully demonstrated. The custom dataset contains only 11 objects, most of which are manually retextured versions of a small number of models, so its scale and diversity are limited. The authors are encouraged to include more materials, shapes, and background conditions, and to further evaluate the method using real images or real sensor environments.

The experimental design and result analysis also need improvement. The paper does not clearly report the number of repetitions for the random-rotation experiments, the random seeds, or any statistical significance tests. Although the manuscript uses expressions such as “significantly increases,” it reports only means and partial standard errors. The LTP distance metric and tolerance parameters also appear to have been selected using the same dataset as the final evaluation, which may introduce tuning bias. The authors should separate the parameter-selection set from the final test set. In addition, “avg_convergence_rate” actually refers to the average number of steps to convergence, so the metric should be renamed, and the missing error information in Table 3 should be added.

The manuscript also contains several formatting and language issues. The headers still include template text such as “Authors Suppressed Due to Excessive Length” and “Title Suppressed Due to Excessive Length,” and a duplicated bullet point appears in the list of encoding schemes. The spellings “Bhattacharya” and “Bhattacharyya” are used inconsistently, and some sentences are overly long or grammatically incorrect.
----------- CSR -----------
SELECTION: no


----------------------- REVIEW 3 ---------------------

SUBMISSION: 96
TITLE: Texture Feature Enhanced Sensory Module for Improved Object Recognition in a Neocortex-inspired Embodied Sensorimotor Agent (Monty)

----------- Scope and Relevance -----------
SCORE: 3 (Must be improved)
----------- Novelty and Significance -----------
SCORE: 4 (Can be improved)
----------- Rigor and Soundness -----------
SCORE: 4 (Can be improved)
----------- Clarity, Style, Completeness, English -----------
SCORE: 4 (Can be improved)
----------- Overall evaluation -----------
SCORE: 1 (weak accept)
----- TEXT:
The paper is rather interesting. The main theme of the paper is analysis of enhanced sensory module for improved object recognition. This theme is indirectly related to topics of the conference. However, the theme could be interesting to some participants of the conference, so the paper can be accepted.

There are small imperfections in the paper text. The text of the paper should be check attentively.
----------- CSR -----------
SELECTION: no
