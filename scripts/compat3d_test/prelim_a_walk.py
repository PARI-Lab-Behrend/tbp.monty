# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""One Prelim A surface walk: per-step patch statistics from a real episode.

Runs any (single-object) experiment with ``show_sensor_output`` and the plotting
switched off, and records from every step's raw patch observation:

* whether the patch centre is on the object (0 < depth < ``ON_OBJECT_MAX_DEPTH``),
* mean HSV value (brightness, 0-1) and grayscale std (texture contrast, 0-255)
  over on-object pixels,

plus a patch RGB thumbnail every ``THUMB_EVERY`` steps. Writes ``<OUT>.json``
and ``<OUT>_thumbs.npz``; with ``PRELIM_A_SAVE_PATCHES=1`` also every step's
patch RGB to ``<OUT>_patches.npz``. Driven by ``prelim_a_checks.py`` and
``select_dataset_shapes.py``.

Usage::

    PRELIM_A_OUT=/tmp/obj_amb python scripts/compat3d_test/prelim_a_walk.py \
        experiment=... <hydra overrides>
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

from tbp.monty.frameworks.run_env import setup_env  # noqa: E402

setup_env()

import sys  # noqa: E402

import numpy as np  # noqa: E402
from skimage.color import rgb2gray, rgb2hsv  # noqa: E402

from tbp.monty.frameworks.run import main  # noqa: E402
from tbp.monty.frameworks.utils import live_plotter as lp  # noqa: E402

ON_OBJECT_MAX_DEPTH = 0.5
THUMB_EVERY = 25

OUT = os.environ["PRELIM_A_OUT"]
SAVE_PATCHES = os.environ.get("PRELIM_A_SAVE_PATCHES") == "1"
_steps = []
_thumbs = []
_patches = []
_hardcoded = lp.LivePlotter.hardcoded_assumptions


def _record(self, observation, model):
    for agent_obs in observation.values():
        if "patch" not in agent_obs:
            continue
        rgb = np.asarray(agent_obs["patch"]["rgba"], np.uint8)[..., :3]
        depth = np.asarray(agent_obs["patch"]["depth"], np.float64)
        on = (depth > 0) & (depth < ON_OBJECT_MAX_DEPTH)
        c = depth.shape[0] // 2
        row = {"center_on_object": bool(on[c, c]), "frac_pixels_on": float(on.mean())}
        if on.any():
            row["v_mean"] = float(rgb2hsv(rgb)[..., 2][on].mean())
            row["gray_std"] = float((rgb2gray(rgb) * 255.0)[on].std())
        _steps.append(row)
        if SAVE_PATCHES:
            _patches.append(rgb)
        if len(_steps) % THUMB_EVERY == 0:
            _thumbs.append(rgb)
    return _hardcoded(self, observation, model)


def _save():
    with open(f"{OUT}.json", "w") as f:
        json.dump(_steps, f)
    if _thumbs:
        np.savez_compressed(f"{OUT}_thumbs.npz", thumbs=np.stack(_thumbs))
    if _patches:
        np.savez_compressed(f"{OUT}_patches.npz", patches=np.stack(_patches))


if __name__ == "__main__":
    lp.LivePlotter.initialize_online_plotting = lambda self: None
    lp.LivePlotter.hardcoded_assumptions = _record
    lp.LivePlotter.show_observations = lambda self, *a, **k: None
    sys.argv.append("experiment.config.show_sensor_output=true")
    try:
        main()
    finally:
        _save()
