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
    collect_distances,
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


class CollectDistancesTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        # Two shapes x two variants, 3 nodes each, all features distinct.
        self.hists = {
            name: self.rng.random((3, 4))
            for name in ("a_000_c0_x", "a_000_c1_y", "b_111_c0_x", "b_111_c1_y")
        }

    def test_within_group_excludes_query_and_its_object(self):
        group_of = {name: name[:5] for name in self.hists}
        dists = collect_distances(
            self.hists, _l1, self.rng, n_queries=12, group_of=group_of
        )
        # Per query: 2 other nodes of its object, 3 of the other variant of its
        # shape, 6 of the other shape.
        self.assertEqual(dists["within_object"].size, 12 * 2)
        self.assertEqual(dists["within_group"].size, 12 * 3)
        self.assertEqual(dists["across_group"].size, 12 * 6)
        self.assertTrue(np.all(dists["within_group"] > 0))

    def test_singleton_groups_give_empty_within_group(self):
        group_of = {name: name for name in self.hists}
        dists = collect_distances(
            self.hists, _l1, self.rng, n_queries=12, group_of=group_of
        )
        self.assertEqual(dists["within_group"].size, 0)
