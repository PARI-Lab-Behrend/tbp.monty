"""Render textured YCB / retextured-YCB objects and assemble labeled montages.

Loads each object's textured GLB into a headless habitat-sim instance, frames it
with a bounding-box-based camera, renders an RGB image, and tiles all objects
into a single labeled montage.

The object name -> mesh-folder mapping strips the leading numeric prefix from the
folder name (e.g. ``006_mustard_bottle`` -> ``mustard_bottle``,
``063-a_marbles`` -> ``a_marbles``).

Usage:
    # 11 retextured objects
    python scripts/visualize_object_datasets.py \
        --dataset retextured --out retextured11_objects.png --cols 4

    # 77 YCB objects
    python scripts/visualize_object_datasets.py \
        --dataset ycb77 --out ycb77_objects.png --cols 8
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import habitat_sim
import magnum as mn
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]

DATASETS = {
    "retextured": {
        "mesh_dir": Path.home() / "tbp/data/retextured_ycb/meshes",
        "config_dir": Path.home() / "tbp/data/retextured_ycb/configs",
        "names_yaml": REPO
        / "src/tbp/monty/conf/env_interface/train_retexturedobj_predefined.yaml",
        "names_key": ("train_env_interface_args", "object_names"),
        "title": "11 retextured objects (retextured_ycb)",
    },
    "ycb77": {
        "mesh_dir": Path.home() / "tbp/data/habitat/versioned_data/ycb_1.2/meshes",
        "config_dir": Path.home() / "tbp/data/habitat/versioned_data/ycb_1.2/configs",
        "names_yaml": REPO
        / "src/tbp/monty/conf/env_interface/eval_77obj_predefined.yaml",
        "names_key": ("eval_env_interface_args", "object_names"),
        "title": "77 YCB objects (habitat ycb_1.2)",
    },
}


def strip_prefix(folder: str) -> str:
    """``006_mustard_bottle`` -> ``mustard_bottle``, ``063-a_marbles`` -> ``a_marbles``."""
    return re.sub(r"^\d+[-_]", "", folder)


def load_object_names(cfg: dict) -> list[str]:
    data = yaml.safe_load(Path(cfg["names_yaml"]).read_text())
    for key in cfg["names_key"]:
        data = data[key]
    return data


def make_sim(resolution: int) -> habitat_sim.Simulator:
    """Create a headless single-camera simulator with a soft 3-point light rig."""
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = "NONE"  # empty scene, just our object
    backend.enable_physics = False
    # These objects set requires_lighting=true (Phong shading), so we need real
    # lights rather than the flat default for an empty scene.
    backend.override_scene_light_defaults = True
    backend.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [resolution, resolution]
    rgb_spec.position = [0.0, 0.0, 0.0]
    rgb_spec.hfov = 60.0

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))

    # The default lighting key is very bright and blows textured surfaces out to
    # white. Replace it with a dimmer camera-relative 3-point rig so the diffuse
    # textures stay visible without clipping.
    from habitat_sim.gfx import LightInfo, LightPositionModel

    cam = LightPositionModel.Camera
    # These GLBs are matte diffuse (metallic=0). A habitat directional LightInfo's
    # `vector` points TOWARD the light source, so to light the camera-facing
    # hemisphere the light must sit on the camera side (+z). A soft camera-relative
    # 3-point rig keyed from the front gives even, well-exposed thumbnails.
    k = 3.0
    lights = [
        # Key: head-on from the camera lights the front hemisphere evenly.
        LightInfo(vector=[0.0, 0.0, 1.0, 0.0], color=[k, k, k], model=cam),
        # Fill from upper-left and lower-right to round out the form.
        LightInfo(vector=[-0.6, 0.5, 0.8, 0.0], color=[k * 0.6, k * 0.6, k * 0.6], model=cam),
        LightInfo(vector=[0.6, -0.4, 0.8, 0.0], color=[k * 0.5, k * 0.5, k * 0.5], model=cam),
    ]
    sim.set_light_setup(lights, habitat_sim.gfx.DEFAULT_LIGHTING_KEY)
    return sim


def render_object(sim: habitat_sim.Simulator, obj_handle: str) -> np.ndarray:
    """Add a single object, frame it, render an RGB image, then remove it."""
    rigid_mgr = sim.get_rigid_object_manager()
    obj = rigid_mgr.add_object_by_template_handle(obj_handle)
    obj.translation = mn.Vector3d(0.0, 0.0, 0.0)

    aabb = obj.root_scene_node.cumulative_bb
    center = obj.translation + mn.Vector3d(aabb.center())
    extent = max(aabb.size_x(), aabb.size_y(), aabb.size_z())

    hfov_rad = math.radians(60.0)
    distance = (extent * 0.5) / math.tan(hfov_rad / 2.0) * 1.8

    # Slightly raised 3/4 view for a nicer silhouette.
    eye = mn.Vector3d(
        center.x + distance * 0.4,
        center.y + distance * 0.3,
        center.z + distance,
    )

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.array([eye.x, eye.y, eye.z], dtype=np.float32)
    forward = (center - eye).normalized()
    world_up = mn.Vector3d(0.0, 1.0, 0.0)
    right = mn.math.cross(forward, world_up).normalized()
    up = mn.math.cross(right, forward).normalized()
    rot = mn.Quaternion.from_matrix(mn.Matrix3x3(right, up, -forward))
    state.rotation = np.quaternion(rot.scalar, *rot.vector)
    agent.set_state(state)

    rgb = sim.get_sensor_observations()["color"][..., :3]
    rigid_mgr.remove_all_objects()
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--cols", type=int, default=8)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    mesh_dir = Path(cfg["mesh_dir"]).expanduser()
    config_dir = Path(cfg["config_dir"]).expanduser()
    object_names = load_object_names(cfg)

    folder_by_name = {
        strip_prefix(p.name): p.name for p in mesh_dir.iterdir() if p.is_dir()
    }

    sim = make_sim(args.resolution)
    obj_mgr = sim.get_object_template_manager()
    # Load every object config in the dataset once; handles keyed by folder name.
    obj_mgr.load_configs(str(config_dir), save_as_defaults=True)

    images: dict[str, np.ndarray] = {}
    for name in object_names:
        folder = folder_by_name[name]
        # Handles are the object_config.json file paths; match the exact folder
        # basename to avoid substring collisions (e.g. baseball vs porous_baseball).
        handles = obj_mgr.get_template_handles(folder)
        handle = next(
            h for h in handles if Path(h).name == f"{folder}.object_config.json"
        )
        images[name] = render_object(sim, handle)
        print(f"rendered {name}")

    sim.close()

    n = len(object_names)
    cols = args.cols
    rows = math.ceil(n / cols)
    # Extra vertical room per row so each title clears the image in the row above.
    row_h = 3.0
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.0, rows * row_h),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, object_names):
        ax.imshow(images[name])
        # pad lifts the title clear of its own image; hspace above gives it room.
        ax.set_title(name, fontsize=9, pad=4)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{cfg['title']} — {n} objects", fontsize=14, y=0.997)
    # Reserve headroom for the suptitle; keep the row hspace set above (don't let
    # tight_layout recompute and re-tighten it).
    fig.subplots_adjust(top=1 - 0.4 / (rows * row_h))
    out_path = Path(args.out).expanduser().absolute()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved montage to {out_path}")


if __name__ == "__main__":
    main()
