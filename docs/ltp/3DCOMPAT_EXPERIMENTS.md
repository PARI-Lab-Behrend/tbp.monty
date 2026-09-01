# 3DCoMPaT++ Experiments: Config Guide & Troubleshooting

How to run the 3DCoMPaT++ experiments on this branch
(`Add-rotation-invariant-LTP-feature-extraction`), how the configs fit
together, and how to recognize and fix the issues this dataset runs into.

The goal of these experiments: test rotation-invariant LTP (local ternary
pattern) texture features on objects that share geometry but differ in
material — 3DCoMPaT++ provides exactly that via "compositions" (the same shape
with different materials assigned to its parts).

## The experiments

```bash
export MONTY_DATA=$PWD/data   # or wherever compat3d_habitat lives

# Pretraining (baseline, then with LTP features)
python run.py experiment=supervised_pre_training_3dcompat_obj
python run.py experiment=supervised_pre_training_3dcompat_obj_ltp

# Eval (needs the baseline pretraining model; see "Eval model path" below)
python run.py experiment=base_config_3dcompatobj_dist_agent
```

Quick smoke test: append `experiment.config.n_train_epochs=1` (one rotation
per object, ~20 s for all 11 objects).

## Config map

Experiments compose Hydra config groups. The 3DCoMPaT-specific files
(all under `src/tbp/monty/conf/`):

| File | Role |
|---|---|
| `experiment/supervised_pre_training_3dcompat_obj.yaml` | Baseline pretraining; mirrors `supervised_pre_training_base` (YCB) |
| `experiment/supervised_pre_training_3dcompat_obj_ltp.yaml` | Same, but with LTP sensor module + learning module |
| `experiment/base_config_3dcompatobj_dist_agent.yaml` | Evidence-LM eval on the pretrained model |
| `environment/habitat_3dcompat_dist_agent_semantics0.yaml` | Distant agent + `data_path: $MONTY_DATA/compat3d_habitat`; copy of `habitat_ycb_dist_agent_semantics0` with only `data_path` changed |
| `env_interface/train_3dcompatobj_predefined.yaml` | The 11 object names + per-epoch rotations (train) |
| `env_interface/eval_3dcompatobj_predefined.yaml` | Same for eval |
| `env_interface/positioning_procedures_{train,eval}/getgoodview_viewfinder_patch_dist01.yaml` | GetGoodView with `good_view_distance: 0.1` — **required for this dataset**, see Issue 2 |

LTP-specific configs the `_ltp` experiment pulls in (shared with the
retextured-YCB experiments): `monty/sensor_module/camera_dist_ltp.yaml`
(adds `ltp` to features + `ltp_config`) and
`monty/learning_module/displacement_delta_d0001_ltp.yaml` (adds `ltp` to
`graph_delta_thresholds`).

To change which objects are used: edit the `object_names` lists in **both**
env interface files. The names must match the directories in
`$MONTY_DATA/compat3d_habitat/objects/` (the converter writes the current
list to `object_names.yaml` next to `objects/`).

## The dataset pipeline

1. 3DCoMPaT++ requires a licensing form before download
   (https://3dcompat-dataset.org/doc/dl-dataset.html); fetch the `3D_ZIP`
   modality.
2. Convert with `~/Projects/compat3d_prep/convert_compat3d_to_habitat.py`
   (outside this repo on purpose — it's tooling for licensed data). It
   applies material compositions, rescales meshes to 0.12 m max extent,
   centers them, and writes the Habitat layout below.
3. Object naming: `<class>_<shape_num>_c<comp>`, e.g. shape `29_000`
   (class `0x29` = vase) in composition 0 → `vase_000_c00`. Current set:
   vase and jug in 4 compositions each (texture discrimination on identical
   geometry) + planter, stool, basket.

Expected layout — the `objects/` parent directory is load-bearing:

```
$MONTY_DATA/compat3d_habitat/
├── object_names.yaml
└── objects/
    ├── vase_000_c00/
    │   ├── vase_000_c00.glb
    │   └── vase_000_c00.object_config.json
    └── ...
```

Keep the dataset out of git: it's licensed data and this repo is public.
(In this clone, `data/` is ignored via `.git/info/exclude` only.)

## Known issues and fixes

### 1. `ValueError: No valid habitat data found in <data_path>`

Habitat's `load_configs` does not recurse, and Monty's `HabitatSim`
(`src/tbp/monty/simulators/habitat/simulator.py`, ~lines 158–201) accepts
only three layouts: `configs/*.object_config.json` (YCB v1.2 style),
`objects/<name>/<name>.object_config.json`, or all config files directly in
`data_path`. Per-object subdirectories **without** the `objects/` parent
match none of them and silently load 0 templates.

Fix: `mkdir objects && mv <object dirs> objects/` (current converter already
emits this layout).

### 2. `ValueError: May be initializing experiment with no visible target object`

The most confusing failure: it appears at a *different episode each run*
(nondeterministic), so it looks like a flaky object or data bug. It isn't.

Cause: `GetGoodView` positioning approaches the object in 1 cm steps until
the closest visible point is within `good_view_distance` (default **0.03 m**).
Furniture rescaled to 0.12 m has sub-centimeter-thin parts (stool legs,
basket weave); during the approach the camera steps through a gap, ends up
*inside* the object seeing nothing, and the procedure raises. Small jitter
in the depth-based semantic estimate decides which episode trips first.

Fix: use the `getgoodview_viewfinder_patch_dist01` positioning configs
(`good_view_distance: 0.1`), already wired into all three experiments. All
11 objects were verified to position safely at 0.1 m. YCB never hits this
because its objects are solid and chunky.

### 3. Object name lookup is substring-based

`HabitatSim.add_object` resolves names with
`get_template_handles(name)[0]` — the *first substring match*. Keep names
non-overlapping (the fixed-width `<class>_<num>_c<NN>` convention is safe;
a hypothetical `vase_1` + `vase_10` pair is not).

### 4. Sparse graphs / dim renders

- With the 0.1 m standoff (Issue 2), one pretraining epoch yields ~35–67
  graph nodes per object (vs ~200+ when the camera can get to 0.03 m). If
  you need denser graphs, regenerate the dataset larger
  (`--target-size 0.2`) so the camera can approach closer relative to
  feature thickness — rather than lowering `good_view_distance` back down.
- Habitat's default lighting renders these PBR materials dim. Usable for
  depth/curvature/LTP, but check renders before blaming the data: load the
  GLB with trimesh, or grab an RGBA observation (snippet below).

### 5. Config snapshot tests

Every yaml in `conf/experiment/` is snapshot-tested (`tests/conf/`). On the
first test run a missing snapshot is auto-created; commit it. If you *edit*
an experiment (or anything it composes), tests fail until you regenerate:
`python src/tbp/monty/conf/update_snapshots.py`, then commit the diff.

Pre-existing on this branch (not caused by the 3DCoMPaT configs): ~58
snapshot tests fail because `retextured_pretrained_dir` was added to
`constants/defaults.yaml` without regenerating snapshots. Running
`update_snapshots.py` once and committing clears it — but it touches every
snapshot file, so do it as its own commit.

### 6. Eval model path

`base_config_3dcompatobj_dist_agent` inlines its model path
(`$MONTY_MODELS/my_trained_models/supervised_pre_training_3dcompat_obj/pretrained/`)
instead of adding a constant — adding a constant changes the resolved config
of *every* experiment and rewrites all snapshots (see Issue 5). Run the
baseline pretraining first, or override `experiment.config.model_name_or_path`.

## Quick verification snippets

Dataset loads and renders (run from the repo root):

```python
from tbp.monty.simulators.habitat import HabitatSim, SingleSensorAgent
agent = SingleSensorAgent(agent_id="agent", sensor_id="cam", resolution=(64, 64))
with HabitatSim(agents=[agent], data_path="data/compat3d_habitat") as sim:
    sim.add_object(name="vase_000_c00", position=(0.0, 1.5, -0.2))
    depth = sim.observations["agent"]["cam"]["depth"]
    print("visible px:", ((depth > 0.01) & (depth < 1.0)).sum())
```

Pretrained model contains the expected graphs (and `ltp` features):

```python
import torch
sd = torch.load("<output_dir>/pretrained/model.pt", weights_only=False)
graphs = sd["lm_dict"][0]["graph_memory"]
for obj, g in graphs.items():
    for ch, graph in g.items():
        print(obj, graph.num_nodes, "nodes;", "ltp" in str(graph.feature_mapping))
```
