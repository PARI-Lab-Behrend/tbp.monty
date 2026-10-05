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

from tbp.monty.frameworks.utils.sensor_processing import (
    get_gabor_hsv_texture_feature_vector,
    get_gabor_rgb_texture_feature_vector,
    get_gabor_texture_feature_vector,
)


def stripes(size=64, wavelength=8.0, vertical=True, phase=0.0):
    x = np.arange(size, dtype=np.float64)
    wave = 0.5 + 0.5 * np.sin(2 * np.pi * x / wavelength + phase)
    return np.tile(wave, (size, 1)) if vertical else np.tile(wave[:, None], (1, size))


class GaborTextureFeatureTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.noise = self.rng.random((64, 64))

    def test_default_length_and_unit_norm(self):
        v = get_gabor_texture_feature_vector(self.noise)
        self.assertEqual(v.shape, (24,))
        self.assertEqual(v.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_fft_and_spatial_paths_agree(self):
        fft = get_gabor_texture_feature_vector(self.noise, {"method": "fft"})
        spatial = get_gabor_texture_feature_vector(self.noise, {"method": "spatial"})
        np.testing.assert_allclose(fft, spatial, atol=1e-5)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            get_gabor_texture_feature_vector(self.noise, {"method": "naive"})

    def test_orientation_is_discriminated(self):
        vertical = get_gabor_texture_feature_vector(stripes(vertical=True))
        horizontal = get_gabor_texture_feature_vector(stripes(vertical=False))
        self.assertGreater(float(np.linalg.norm(vertical - horizontal)), 0.3)

    def test_insensitive_to_phase_shift(self):
        # Magnitudes are shift-invariant except near the border of a 64-px patch
        # filtered with kernels up to 55 px wide, so the phase effect is small
        # *relative to* what the descriptor is meant to discriminate.
        a = get_gabor_texture_feature_vector(stripes(phase=0.0))
        b = get_gabor_texture_feature_vector(stripes(phase=1.3))
        rotated = get_gabor_texture_feature_vector(stripes(vertical=False))
        self.assertLess(
            float(np.linalg.norm(a - b)), 0.1 * float(np.linalg.norm(a - rotated))
        )

    def test_wavelength_is_discriminated(self):
        fine = get_gabor_texture_feature_vector(stripes(wavelength=4.0))
        coarse = get_gabor_texture_feature_vector(stripes(wavelength=16.0))
        self.assertGreater(float(np.linalg.norm(fine - coarse)), 0.3)

    def test_uint8_input_is_scaled(self):
        as_uint8 = (self.noise * 255).round().astype(np.uint8)
        np.testing.assert_allclose(
            get_gabor_texture_feature_vector(as_uint8),
            get_gabor_texture_feature_vector(as_uint8 / 255.0),
            atol=1e-6,
        )

    def test_empty_mask_gives_zero_vector(self):
        v = get_gabor_texture_feature_vector(self.noise, mask=np.zeros((64, 64), bool))
        np.testing.assert_array_equal(v, np.zeros(24, np.float32))

    def test_mask_restricts_statistics(self):
        image = np.hstack([stripes(vertical=True)[:, :32], stripes(vertical=False)[:, 32:]])
        left = np.zeros((64, 64), bool)
        left[:, :32] = True
        masked = get_gabor_texture_feature_vector(image, mask=left)
        full = get_gabor_texture_feature_vector(image)
        self.assertGreater(float(np.linalg.norm(masked - full)), 0.01)

    def test_custom_bank_size(self):
        v = get_gabor_texture_feature_vector(
            self.noise, {"wavelengths": [4, 8], "n_orientations": 6}
        )
        self.assertEqual(v.shape, (24,))  # 2 x 6 filters x (mean, std)


class GaborColorTextureFeatureTest(unittest.TestCase):
    def setUp(self):
        self.rgb = (np.random.default_rng(1).random((64, 64, 4)) * 255).astype(np.uint8)

    def test_rgb_length_and_per_channel_norm(self):
        v = get_gabor_rgb_texture_feature_vector(self.rgb)
        self.assertEqual(v.shape, (72,))
        for c in range(3):
            self.assertAlmostEqual(float(np.linalg.norm(v[24 * c : 24 * (c + 1)])), 1.0, 5)

    def test_rgb_channels_match_single_channel_descriptor(self):
        v = get_gabor_rgb_texture_feature_vector(self.rgb)
        np.testing.assert_allclose(
            v[24:48], get_gabor_texture_feature_vector(self.rgb[:, :, 1]), atol=1e-6
        )

    def test_hsv_length(self):
        self.assertEqual(get_gabor_hsv_texture_feature_vector(self.rgb).shape, (96,))

    def test_hsv_hue_channels_are_zero_for_gray_patch(self):
        gray = np.repeat(self.rgb[:, :, :1], 3, axis=2)
        v = get_gabor_hsv_texture_feature_vector(gray)
        # Saturation is 0 everywhere, so S*cos(H), S*sin(H) and S carry no energy.
        np.testing.assert_array_equal(v[:72], np.zeros(72, np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(v[72:])), 1.0, 5)

    def test_too_few_channels_raise(self):
        with self.assertRaises(ValueError):
            get_gabor_rgb_texture_feature_vector(self.rgb[:, :, :2])
        with self.assertRaises(ValueError):
            get_gabor_hsv_texture_feature_vector(self.rgb[:, :, :2])


if __name__ == "__main__":
    unittest.main()
