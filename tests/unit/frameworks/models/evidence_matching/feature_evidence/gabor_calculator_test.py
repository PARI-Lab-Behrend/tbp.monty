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

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
)


def unit_rows(rng, n_rows, n_channels, width=24):
    v = rng.random((n_rows, n_channels, width))
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    return v.reshape(n_rows, n_channels * width)


class GaborFeatureEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def _calculate(self, feature, stored, query, tolerance, extra=None):
        order = [feature] + (list(extra) if extra else [])
        array = stored if not extra else np.hstack([stored] + [extra[k][0] for k in extra])
        query_features = {feature: query, **({k: extra[k][1] for k in extra} if extra else {})}
        return DefaultFeatureEvidenceCalculator.calculate(
            channel_feature_array=array,
            channel_feature_order=order,
            channel_feature_weights={k: 1.0 for k in order},
            channel_query_features=query_features,
            channel_tolerances={k: tolerance for k in order},
        )

    def test_rms_of_per_channel_euclidean_distances(self):
        for feature, n_channels in (("gabor", 1), ("gabor_rgb", 3), ("gabor_hsv", 4)):
            stored = unit_rows(self.rng, 6, n_channels)
            query = unit_rows(self.rng, 1, n_channels)[0]
            tolerance = 0.5
            evidence = self._calculate(feature, stored, query, tolerance)
            per_channel = np.linalg.norm(
                (stored - query).reshape(6, n_channels, -1), axis=2
            )
            rms = np.sqrt((per_channel**2).mean(axis=1))
            expected = np.clip(1.0 - rms / tolerance, 0.0, None)
            np.testing.assert_allclose(evidence, expected, atol=1e-9)

    def test_identical_vector_gets_full_evidence(self):
        stored = unit_rows(self.rng, 3, 3)
        evidence = self._calculate("gabor_rgb", stored, stored[1], 0.3)
        self.assertAlmostEqual(float(evidence[1]), 1.0)

    def test_vector_counts_as_one_feature_in_the_weighted_average(self):
        # A 72-value gabor_rgb vector must weigh like one feature, not 72: with a
        # perfectly matching scalar feature beside it, evidence is the mean of the two.
        stored = unit_rows(self.rng, 1, 3)
        query = unit_rows(self.rng, 1, 3)[0]
        gabor_only = self._calculate("gabor_rgb", stored, query, 2.0)
        both = self._calculate(
            "gabor_rgb", stored, query, 2.0,
            extra={"curvature": (np.array([[0.5]]), 0.5)},
        )
        np.testing.assert_allclose(both, (gabor_only + 1.0) / 2.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
