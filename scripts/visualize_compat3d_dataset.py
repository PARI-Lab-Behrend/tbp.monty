"""Render the compat3d_habitat objects with textures and assemble a labeled grid.

Loads each object's GLB (with embedded textures) into a headless habitat-sim
instance, frames it with a camera based on its bounding box, renders an RGB
image, and tiles all objects into a single labeled montage.

Usage:
    python scripts/visualize_compat3d_dataset.py \
        --data-path data/compat3d_habitat \
        --out compat3d_dataset.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import habitat_sim
import magnum as mn
import matplotlib.pyplot as plt
import numpy as np
import yaml


def make_sim(resolution: int) -> habitat_sim.Simulator:
    """Create a headless single-camera simulator with default lighting."""
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = "NONE"  # empty scene, just our object
    backend.enable_physics = False
    # Objects in this dataset set requires_lighting=true (Phong shading), so we
    # need real lights rather than the flat default for an empty scene.
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

    # The default lighting key is very bright and blows the textured surfaces out
    # to white. Replace it with a dimmer 3-point rig (camera-relative directional
    # lights) so the diffuse textures stay visible without clipping.
    from habitat_sim.gfx import LightInfo, LightPositionModel

    cam = LightPositionModel.Camera
    # NOTE: the GLBs in this dataset carry no textures, no images and no
    # per-material colors (empty materials with just part names), so there is no
    # diffuse albedo to show. These lights exist only to make the bare geometry
    # legible as shaded 3D form. See the script docstring / README note.
    lights = [
        # Strong frontal light (aligned with the camera) lights camera-facing
        # surfaces; angled key/fill give the shape its 3D form.
        LightInfo(vector=[0.0, 0.0, -1.0, 0.0], color=[4.0, 4.0, 4.0], model=cam),
        LightInfo(vector=[0.6, 0.7, -0.6, 0.0], color=[3.0, 3.0, 3.0], model=cam),
        LightInfo(vector=[-0.7, 0.2, -0.5, 0.0], color=[1.6, 1.6, 1.6], model=cam),
    ]
    sim.set_light_setup(lights, habitat_sim.gfx.DEFAULT_LIGHTING_KEY)
    return sim


def render_object(sim: habitat_sim.Simulator, obj_handle: str) -> np.ndarray:
    """Add a single object, frame it, render an RGB image, then remove it."""
    rigid_mgr = sim.get_rigid_object_manager()
    obj = rigid_mgr.add_object_by_template_handle(obj_handle)
    obj.translation = mn.Vector3d(0.0, 0.0, 0.0)

    # World-space axis-aligned bounding box to compute object center and size.
    aabb = obj.root_scene_node.cumulative_bb
    center = obj.translation + mn.Vector3d(aabb.center())
    extent = max(aabb.size_x(), aabb.size_y(), aabb.size_z())

    # Distance so the object comfortably fills the 60-deg FOV frame.
    hfov_rad = math.radians(60.0)
    distance = (extent * 0.5) / math.tan(hfov_rad / 2.0) * 1.8

    # Look at the object from a slightly raised 3/4 view for a nicer silhouette.
    eye = mn.Vector3d(
        center.x + distance * 0.4,
        center.y + distance * 0.3,
        center.z + distance,
    )

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.array([eye.x, eye.y, eye.z], dtype=np.float32)
    # Orient the camera to look at the object center.
    forward = (center - eye).normalized()
    world_up = mn.Vector3d(0.0, 1.0, 0.0)
    right = mn.math.cross(forward, world_up).normalized()
    up = mn.math.cross(right, forward).normalized()
    rot = mn.Quaternion.from_matrix(
        mn.Matrix3x3(right, up, -forward)  # camera looks down -z
    )
    state.rotation = np.quaternion(rot.scalar, *rot.vector)
    agent.set_state(state)

    rgb = sim.get_sensor_observations()["color"][..., :3]
    rigid_mgr.remove_all_objects()
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/compat3d_habitat")
    parser.add_argument("--out", default="compat3d_dataset.png")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser().absolute()
    names_file = data_path / "object_names.yaml"
    object_names = yaml.safe_load(names_file.read_text())["object_names"]

    sim = make_sim(args.resolution)
    obj_mgr = sim.get_object_template_manager()

    images: dict[str, np.ndarray] = {}
    for name in object_names:
        config_dir = data_path / "objects" / name
        obj_mgr.load_configs(str(config_dir), save_as_defaults=True)
        handle = obj_mgr.get_template_handles(name)[0]
        images[name] = render_object(sim, handle)
        print(f"rendered {name}")

    sim.close()

    n = len(object_names)
    cols = args.cols
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, object_names):
        ax.imshow(images[name])
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"compat3d_habitat ({n} objects) — geometry only; GLBs carry no textures",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = Path(args.out).expanduser().absolute()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved montage to {out_path}")


if __name__ == "__main__":
    main()
