# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
r"""Run an experiment and record per-step wall time per episode (Phase 6, level 2).

Wraps, with ``time.perf_counter``:

- ``MontyBase.step`` and ``MontyBase.motor_only_step``: one Monty step each
  (sensor modules + motor system; ``step`` also runs the learning modules).
  The eval loop calls ``motor_only_step`` when the surface agent is off the
  surface or re-orienting, so both together are every Monty step
  (``monty_steps`` in the eval stats).
- ``CameraSM.step``: the sensor module, of which
  ``ObservationProcessor.process`` is the feature extraction.
- ``GraphLM.matching_step``: learning-module matching (evidence update).

``run_episode`` (of ``MontyExperiment`` and every subclass that overrides it;
only the outermost call records) delimits episodes. One JSON line per episode is
appended to ``$STEP_TIMING_OUT``: totals and call counts for each wrapped
method, the median and p95 per Monty step, and the episode wall time
(which also includes the simulator and logging).

Runs through ``run.py`` (serial), not ``run_parallel.py``: parallel workers are
fresh processes that would not inherit the wrappers. For clean numbers pin one
core and one thread::

    STEP_TIMING_OUT=/tmp/t.jsonl OMP_NUM_THREADS=1 taskset -c 5 \
        python scripts/phase6_step_timing.py experiment=... <overrides>
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path

from tbp.monty.frameworks.run_env import setup_env

setup_env()

import numpy as np  # noqa: E402

from tbp.monty.frameworks.experiments import (  # noqa: E402, F401  (registers subclasses)
    object_recognition_experiments,
    pretraining_experiments,
)
from tbp.monty.frameworks.experiments.monty_experiment import (  # noqa: E402
    MontyExperiment,
)
from tbp.monty.frameworks.models.graph_matching import GraphLM  # noqa: E402
from tbp.monty.frameworks.models.monty_base import MontyBase  # noqa: E402
from tbp.monty.frameworks.models.sensor_modules import (  # noqa: E402
    CameraSM,
    ObservationProcessor,
)
from tbp.monty.frameworks.run import main  # noqa: E402

# Wrapped methods, keyed by the name used in the output.
TIMED = {
    "model_step": (MontyBase, "step"),
    "motor_only_step": (MontyBase, "motor_only_step"),
    "sm_step": (CameraSM, "step"),
    "sm_process": (ObservationProcessor, "process"),
    "lm_matching_step": (GraphLM, "matching_step"),
}

_totals: dict[str, float] = {}
_counts: dict[str, int] = {}
_monty_step_s: list[float] = []  # model_step and motor_only_step durations
_episode_depth = {"n": 0}


def _reset() -> None:
    for key in TIMED:
        _totals[key] = 0.0
        _counts[key] = 0
    _monty_step_s.clear()


def _timed(key: str, method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            _totals[key] += elapsed
            _counts[key] += 1
            if key in ("model_step", "motor_only_step"):
                _monty_step_s.append(elapsed)

    return wrapper


def _run_episode_timed(original, out_path: str):
    @functools.wraps(original)
    def wrapper(self):
        if _episode_depth["n"]:  # a subclass's run_episode calling super()
            return original(self)
        _episode_depth["n"] += 1
        _reset()
        start = time.perf_counter()
        try:
            original(self)
        finally:
            _episode_depth["n"] -= 1
        wall = time.perf_counter() - start
        steps = np.asarray(_monty_step_s)
        record = {
            "mode": self.experiment_mode.value,
            # post_episode already incremented the counter.
            "episode": (
                self.train_episodes - 1
                if self.experiment_mode.value == "train"
                else self.eval_episodes - 1
            ),
            "episode_wall_s": wall,
            "monty_steps": int(steps.size),
            "monty_step_median_ms": (
                float(np.median(steps) * 1e3) if steps.size else None
            ),
            "monty_step_p95_ms": (
                float(np.percentile(steps, 95) * 1e3) if steps.size else None
            ),
        }
        for key in TIMED:
            record[f"{key}_total_s"] = _totals[key]
            record[f"{key}_calls"] = _counts[key]
        with Path(out_path).open("a") as f:
            f.write(json.dumps(record) + "\n")

    return wrapper


def _with_subclasses(cls) -> list[type]:
    found = [cls]
    for sub in cls.__subclasses__():
        found.extend(_with_subclasses(sub))
    return found


def install(out_path: str) -> None:
    for key, (cls, name) in TIMED.items():
        setattr(cls, name, _timed(key, getattr(cls, name)))
    for cls in _with_subclasses(MontyExperiment):
        if "run_episode" in vars(cls):
            cls.run_episode = _run_episode_timed(cls.run_episode, out_path)
    _reset()


if __name__ == "__main__":
    install(os.environ["STEP_TIMING_OUT"])
    main()
