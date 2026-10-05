# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.utils.sensor_processing import (
    get_gabor_rgb_texture_feature_vector,
)
from tests.unit.frameworks.models.camera_sm_ltp_test import (
    UNIFORM_LTP_CONFIG,
    _make_on_object_observation,
)


def _step(sm: CameraSM, obs):
    sm.reset()
    ctx = Mock()
    ctx.rng = np.random.RandomState(0)
    return sm.step(ctx, obs).non_morphological_features


class CameraSMGaborTest(unittest.TestCase):
    def test_emits_all_gabor_features_with_expected_lengths(self) -> None:
        sm = CameraSM(
            sensor_module_id="patch",
            features=["on_object", "gabor", "gabor_rgb", "gabor_hsv"],
        )
        features = _step(sm, _make_on_object_observation())
        self.assertEqual(np.asarray(features["gabor"]).shape, (24,))
        self.assertEqual(np.asarray(features["gabor_rgb"]).shape, (72,))
        self.assertEqual(np.asarray(features["gabor_hsv"]).shape, (96,))

    def test_gabor_rgb_matches_direct_extraction_and_config_is_plumbed(self) -> None:
        obs = _make_on_object_observation(seed=3)
        config = {"wavelengths": [4, 8], "n_orientations": 6}
        sm = CameraSM(
            sensor_module_id="patch", features=["on_object", "gabor_rgb"],
            gabor_config=config,
        )
        np.testing.assert_allclose(
            _step(sm, obs)["gabor_rgb"],
            get_gabor_rgb_texture_feature_vector(obs["rgba"][:, :, :3], config),
            atol=1e-6,
        )

    def test_combined_feature_set_emits_every_texture_feature(self) -> None:
        # The combined-feature pretraining sensor module stores every texture
        # feature at once (JOURNAL_EXPERIMENT_PLAN.md, "Train once").
        keys = ["hsv", "ltp", "ltp_rgb", "ltp_hsv", "gabor", "gabor_rgb", "gabor_hsv"]
        sm = CameraSM(
            sensor_module_id="patch", features=["on_object", *keys],
            ltp_config=UNIFORM_LTP_CONFIG,
        )
        features = _step(sm, _make_on_object_observation())
        for key in keys:
            self.assertIn(key, features)


if __name__ == "__main__":
    unittest.main()
