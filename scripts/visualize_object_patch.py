"""Visualize an object together with the sensor patch, using real habitat renders.

For each object this builds a `HabitatEnvironment` with the same agent parameters as
the retextured-YCB dist/surf environment configs, renders the view finder (a wide
view of the whole object) and the patch (the small, zoomed-in sensor), and lays them
out side by side:

    [ object view-finder, with the patch footprint boxed ]  [ patch RGB ]

The `surf` agent parameters (resolutions [64,64]/[64,64], zooms [10.0, 1.0]) are taken
verbatim from conf/environment/retextured_surf_agent_semantics0.yaml, i.e. the exact
patch resolution and zoom the randrot surf-agent experiments use.

The patch camera points at the view-finder centre, only more zoomed in, so the patch
footprint on the object is the central region whose size is
(view_finder_zoom / patch_zoom) of the view-finder frame.

This deliberately does NOT use plot_utils_analysis / the point-cloud model rendering,
which does not render the actual object surface.

Usage:
    python scripts/visualize_object_patch.py --agent surf \
        --objects 025_mug 011_banana 055_baseball --out object_patch_surf.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from tbp.monty.simulators.habitat import MultiSensorAgent
from tbp.monty.simulators.habitat.environment import HabitatEnvironment

AGENT_ID = "agent_id_0"
OBJECT_POSITION = (0.0, 1.5, -0.1)

#: Agent parameters copied from conf/environment/retextured_{dist,surf}_agent*.yaml.
#: `zooms` = [patch, view_finder]; the patch is the more-zoomed (magnified) sensor.
AGENT_ARGS = {
    "dist": dict(
        agent_id=AGENT_ID,
        sensor_ids=["patch", "view_finder"],
        height=0.0,
        position=(0.0, 1.5, 0.2),
        resolutions=[(64, 64), (256, 256)],
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        rotations=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        semantics=[False, False],
        zooms=[10.0, 1.0],
    ),
    "surf": dict(
        agent_id=AGENT_ID,
        sensor_ids=["patch", "view_finder"],
        height=0.0,
        position=(0.0, 1.5, 0.1),
        resolutions=[(64, 64), (64, 64)],
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.03)],
        rotations=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
        semantics=[False, False],
        zooms=[10.0, 1.0],
        action_space_type="surface_agent",
    ),
}


def render(agent_type: str, obj_name: str, data_path: Path, lights=None) -> dict:
    """Render the view finder and patch (rgba + depth) for one object.

    Returns:
        Dict keyed by sensor id, each holding {"rgba": ..., "depth": ...}.
    """
    env = HabitatEnvironment(
        agents={"agent_type": MultiSensorAgent, "agent_args": AGENT_ARGS[agent_type]},
        objects=[{"name": obj_name, "position": OBJECT_POSITION}],
        data_path=str(data_path),
        lights=lights,
    )
    obs = env.reset()[0][AGENT_ID]
    out = {
        sensor: {
            "rgba": np.asarray(obs[sensor]["rgba"], dtype=np.uint8),
            "depth": np.asarray(obs[sensor]["depth"], dtype=float),
        }
        for sensor in ("view_finder", "patch")
    }
    env.close()
    return out


def patch_box_fraction(agent_type: str) -> float:
    """Fraction of the view-finder frame that the patch footprint spans."""
    patch_zoom, vf_zoom = AGENT_ARGS[agent_type]["zooms"]
    return vf_zoom / patch_zoom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["dist", "surf"], default="surf")
    parser.add_argument("--data-path", default="~/tbp/data/retextured_ycb")
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["025_mug", "011_banana", "024_bowl", "055_baseball"],
    )
    parser.add_argument(
        "--obj-cols", type=int, default=2, help="objects per row in the grid"
    )
    parser.add_argument("--out", default="object_patch.png")
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser()
    frac = patch_box_fraction(args.agent)

    # Grid of objects: `obj_cols` objects per row. Each object occupies two adjacent
    # axis columns: [object view-finder + patch footprint] and [patch rgb].
    obj_cols = args.obj_cols
    obj_rows = math.ceil(len(args.objects) / obj_cols)
    fig, axes = plt.subplots(
        obj_rows,
        obj_cols * 2,
        figsize=(obj_cols * 2 * 2.8, obj_rows * 3.2),
        squeeze=False,
    )
    # Hide every axis up front so unused grid cells stay blank.
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for idx, obj_name in enumerate(args.objects):
        r, c = idx // obj_cols, idx % obj_cols
        obs = render(args.agent, obj_name, data_path)
        vf = obs["view_finder"]["rgba"][..., :3]
        patch_rgb = obs["patch"]["rgba"][..., :3]

        # --- object with patch footprint boxed at the centre ---
        ax = axes[r][c * 2]
        ax.imshow(vf)
        h, w = vf.shape[:2]
        bw, bh = w * frac, h * frac
        ax.add_patch(
            Rectangle(
                ((w - bw) / 2, (h - bh) / 2),
                bw,
                bh,
                fill=False,
                edgecolor="#00b4ff",
                linewidth=2.5,
            )
        )
        ax.set_title(f"{obj_name}\nobject + patch footprint", fontsize=10)

        # --- the patch itself (zoomed RGB) ---
        ax = axes[r][c * 2 + 1]
        ax.imshow(patch_rgb)
        ax.set_title("patch (rgb)", fontsize=10)

        print(f"rendered {args.agent} / {obj_name}")

    fig.suptitle(f"{args.agent} agent — object + patch (habitat render)", fontsize=13)
    fig.tight_layout()
    out_path = Path(args.out).expanduser()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
