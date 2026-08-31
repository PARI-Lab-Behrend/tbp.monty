"""Render habitat views under each lighting mode, for the dist and surf agents.

Builds a `HabitatEnvironment` with the same agent parameters as the retextured-YCB
dist/surf environment configs, then renders the view finder and the patch for each
object under each `lights` setting, so the effect of the lighting can be inspected
side by side.

Usage:
    python scripts/visualize_lighting.py --out-dir lighting_snapshots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tbp.monty.simulators.habitat import MultiSensorAgent
from tbp.monty.simulators.habitat.environment import HabitatEnvironment
from tbp.monty.simulators.habitat.simulator import AMBIENT_LIGHTS

AGENT_ID = "agent_id_0"
OBJECT_POSITION = (0.0, 1.5, -0.1)

#: Agent parameters copied from conf/environment/retextured_{dist,surf}_agent*.yaml
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

LIGHTING_MODES = {
    "default (directional)": None,
    "ambient (6 lights)": AMBIENT_LIGHTS,
    "unlit (flat)": [],
}


def render(agent_type: str, obj_name: str, lights, data_path: Path) -> dict:
    """Render the view finder and patch for one object under one lighting mode.

    Returns:
        The RGB image from each sensor, keyed by sensor id.
    """
    env = HabitatEnvironment(
        agents={"agent_type": MultiSensorAgent, "agent_args": AGENT_ARGS[agent_type]},
        objects=[{"name": obj_name, "position": OBJECT_POSITION}],
        data_path=str(data_path),
        lights=lights,
    )
    obs = env.reset()[0][AGENT_ID]
    images = {
        sensor: obs[sensor]["rgba"][..., :3] for sensor in ("view_finder", "patch")
    }
    env.close()
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="~/tbp/data/retextured_ycb")
    parser.add_argument("--out-dir", default="lighting_snapshots")
    parser.add_argument(
        "--objects", nargs="+", default=["025_mug", "011_banana", "024_bowl"]
    )
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    for agent_type in ("dist", "surf"):
        n_rows = len(args.objects)
        n_cols = 2 * len(LIGHTING_MODES)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows), squeeze=False
        )
        for row, obj_name in enumerate(args.objects):
            for mode_idx, (mode, lights) in enumerate(LIGHTING_MODES.items()):
                images = render(agent_type, obj_name, lights, data_path)
                for offset, sensor in enumerate(("view_finder", "patch")):
                    ax = axes[row][2 * mode_idx + offset]
                    ax.imshow(np.asarray(images[sensor], dtype=np.uint8))
                    ax.axis("off")
                    if row == 0:
                        ax.set_title(f"{mode}\n{sensor}", fontsize=9)
                    if offset == 0 and mode_idx == 0:
                        ax.text(
                            -0.1,
                            0.5,
                            obj_name,
                            transform=ax.transAxes,
                            rotation=90,
                            va="center",
                            ha="center",
                            fontsize=10,
                        )
                print(f"rendered {agent_type} / {obj_name} / {mode}")

        fig.suptitle(f"{agent_type} agent — habitat lighting modes", fontsize=13)
        fig.tight_layout()
        out_path = out_dir / f"lighting_{agent_type}_agent.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
