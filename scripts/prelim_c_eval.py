# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Run an eval experiment and record each feature's own evidence (Prelim C).

Wraps ``DefaultFeatureEvidenceCalculator.calculate``: on every call it also
computes, for each feature with a nonzero configured weight, the evidence that
feature alone produces (same call with every other weight set to 0), and keeps
the max and mean over the episode(s). Writes ``$PRELIM_C_OUT`` (JSON) with
those statistics, the number of calls, and wall time.

Usage::

    PRELIM_C_OUT=/tmp/ev.json python scripts/prelim_c_eval.py experiment=... <overrides>
"""

from __future__ import annotations

import json
import os
import time

from tbp.monty.frameworks.run_env import setup_env

setup_env()

import numpy as np  # noqa: E402

from tbp.monty.frameworks.models.evidence_matching.feature_evidence import (  # noqa: E402
    calculator as calc_module,
)
from tbp.monty.frameworks.run import main  # noqa: E402

Calc = calc_module.DefaultFeatureEvidenceCalculator
_original = Calc.calculate.__func__
_stats = {}  # feature -> {"max", "sum_mean", "n"}
_calls = {"n": 0}


def _nonzero(weight) -> bool:
    return bool(np.any(np.asarray(weight, dtype=float) != 0))


def _calculate(cls, channel_feature_array, channel_feature_order,
               channel_feature_weights, channel_query_features, channel_tolerances):
    result = _original(cls, channel_feature_array, channel_feature_order,
                       channel_feature_weights, channel_query_features,
                       channel_tolerances)
    _calls["n"] += 1
    for feature, weight in channel_feature_weights.items():
        if feature not in channel_query_features or not _nonzero(weight):
            continue
        only = {
            k: (w if k == feature else np.zeros_like(np.asarray(w, dtype=float)))
            for k, w in channel_feature_weights.items()
        }
        ev = _original(cls, channel_feature_array, channel_feature_order, only,
                       channel_query_features, channel_tolerances)
        s = _stats.setdefault(feature, {"max": 0.0, "sum_mean": 0.0, "n": 0})
        s["max"] = max(s["max"], float(np.max(ev)) if ev.size else 0.0)
        s["sum_mean"] += float(np.mean(ev)) if ev.size else 0.0
        s["n"] += 1
    return result


if __name__ == "__main__":
    Calc.calculate = classmethod(_calculate)
    start = time.perf_counter()
    try:
        main()
    finally:
        out = {
            "wall_s": time.perf_counter() - start,
            "calls": _calls["n"],
            "features": {
                k: {"max": v["max"], "mean": v["sum_mean"] / max(v["n"], 1)}
                for k, v in _stats.items()
            },
        }
        with open(os.environ["PRELIM_C_OUT"], "w") as f:
            json.dump(out, f, indent=1)
