# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np

from tbp.monty.frameworks.utils.ltp_distance_analysis import (
    _l1,
    _mann_whitney_auc,
    location_specificity_auc,
)


class MannWhitneyAucTest(unittest.TestCase):
    def test_separated_groups(self):
        self.assertEqual(_mann_whitney_auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])), 1.0)
        self.assertEqual(_mann_whitney_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])), 0.0)

    def test_ties_count_half(self):
        self.assertEqual(_mann_whitney_auc(np.array([1.0, 1.0]), np.array([1.0])), 0.5)

    def test_partial_overlap(self):
        # high > low in 3 of 4 pairs: (2>1), (2<3), (4>1), (4>3)
        self.assertEqual(
            _mann_whitney_auc(np.array([1.0, 3.0]), np.array([2.0, 4.0])), 0.75
        )


class LocationSpecificityAucTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        # 200 nodes along a 20 cm line.
        x = np.linspace(0.0, 0.2, 200)
        self.positions = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)

    def _hists_from(self, values):
        h = np.stack([values, 1.0 - values], axis=1)
        return h / h.sum(axis=1, keepdims=True)

    def test_feature_that_varies_with_location_scores_high(self):
        hists = self._hists_from(np.linspace(0.05, 0.95, 200))
        res = location_specificity_auc(self.positions, hists, _l1, self.rng)
        self.assertGreater(res["auc"], 0.95)
        self.assertGreater(res["n_near"], 0)
        self.assertGreater(res["n_far"], 0)

    def test_location_independent_feature_scores_near_chance(self):
        hists = self._hists_from(self.rng.uniform(0.05, 0.95, 200))
        res = location_specificity_auc(self.positions, hists, _l1, self.rng)
        self.assertAlmostEqual(res["auc"], 0.5, delta=0.05)

    def test_no_far_pairs_returns_nan(self):
        hists = self._hists_from(np.linspace(0.05, 0.95, 200))
        res = location_specificity_auc(
            self.positions, hists, _l1, self.rng, near=0.01, far=1.0
        )
        self.assertTrue(np.isnan(res["auc"]))
        self.assertEqual(res["n_far"], 0)


if __name__ == "__main__":
    unittest.main()
