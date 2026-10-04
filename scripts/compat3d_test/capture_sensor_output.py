# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Run a Monty experiment with ``show_sensor_output`` and save the frames headless.

This is the trusted view of what the sensors see: it runs the real experiment
(surface agent, motor policy, sensor module) and saves the ``LivePlotter``
figure that ``show_sensor_output=true`` would show -- view_finder rgba (with
the plotter's patch box) and patch depth -- every ``FRAME_EVERY`` steps. The
patch rgba from the same observation is pasted on the right, since the plotter
itself only shows patch depth.

Note: the plotter's white box is a fixed 1/10 of the view finder and overstates
the patch. Measured on the 3DCoMPaT surface-agent setup, the patch spans ~2.3 of
the view finder's 64 px (3.6% of its width) and ~3.9 mm of surface, because
Habitat zoom divides the field of view (90 -> 9 deg) and the view finder sits
3 cm behind the patch.

Usage (any experiment config and Hydra overrides; the flag is set here)::

    FRAME_DIR=/tmp/frames FRAME_EVERY=50 python \
        scripts/compat3d_test/capture_sensor_output.py \
        experiment=journal/surf_pre_training_3dcompat11_ltp_rgb \
        experiment.config.n_train_epochs=1 \
        "experiment.config.train_env_interface_args.object_names=[ottoman_000_leather00]"
"""

import os

import matplotlib

matplotlib.use("Agg")

from tbp.monty.frameworks.run_env import setup_env  # noqa: E402

setup_env()

import sys  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tbp.monty.frameworks.run import main  # noqa: E402
from tbp.monty.frameworks.utils import live_plotter as lp  # noqa: E402

OUT = os.environ["FRAME_DIR"]
EVERY = int(os.environ.get("FRAME_EVERY", "25"))

_state = {"episode": -1, "patch": None}
_init = lp.LivePlotter.initialize_online_plotting
_hardcoded = lp.LivePlotter.hardcoded_assumptions
_show = lp.LivePlotter.show_observations


def _initialize(self):
    _state["episode"] += 1
    _init(self)


def _hardcoded_assumptions(self, observation, model):
    for agent_obs in observation.values():
        if "patch" in agent_obs:
            _state["patch"] = np.asarray(agent_obs["patch"]["rgba"], np.uint8)[..., :3]
    return _hardcoded(self, observation, model)


def _show_observations(self, *args, **kwargs):
    _show(self, *args, **kwargs)
    step = args[6] if len(args) > 6 else kwargs["step"]
    if step % EVERY:
        return
    path = f"{OUT}/ep{_state['episode']:02d}_step{step:04d}.png"
    self.fig.savefig(path, dpi=80)
    fig = Image.open(path).convert("RGB")
    patch = Image.fromarray(_state["patch"]).resize(
        (fig.height // 2, fig.height // 2), Image.NEAREST
    )
    frame = Image.new("RGB", (fig.width + patch.width, fig.height), "white")
    frame.paste(fig, (0, 0))
    frame.paste(patch, (fig.width, fig.height // 4))
    frame.save(path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    lp.LivePlotter.initialize_online_plotting = _initialize
    lp.LivePlotter.hardcoded_assumptions = _hardcoded_assumptions
    lp.LivePlotter.show_observations = _show_observations
    sys.argv.append("experiment.config.show_sensor_output=true")
    main()
