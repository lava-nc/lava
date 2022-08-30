# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.utils.weightutils import SignMode, truncate_weights


class TestTruncateWeights(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = np.array(
            [-300, -256, -255, -49, -1, 0, 49, 255, 256, 300]
        )

    def test_truncate_weights_mixed_8_bits(self) -> None:
        """Tests truncation of weights for mixed sign mode."""
        weights_truncated = truncate_weights(
            weights=self.weights,
            sign_mode=SignMode.MIXED.value,
            num_weight_bits=8)
        expected = np.array([-256, -256, -256, -50, -2, 0, 48, 254, 254, 254])

        np.testing.assert_array_equal(weights_truncated, expected)

    def test_truncate_weights_excitatory_8_bits(self) -> None:
        """Tests truncation of weights for mixed sign mode."""
        weights_truncated = truncate_weights(
            weights=self.weights,
            sign_mode=SignMode.EXCITATORY.value,
            num_weight_bits=8)
        expected = np.array([0, 0, 0, 0, 0, 0, 49, 255, 255, 255])

        np.testing.assert_array_equal(weights_truncated, expected)

    def test_truncate_weights_inhibitory_8_bits(self) -> None:
        """Tests truncation of weights for mixed sign mode."""
        weights_truncated = truncate_weights(
            weights=self.weights,
            sign_mode=SignMode.INHIBITORY.value,
            num_weight_bits=8)
        expected = np.array([-256, -256, -255, -49, -1, 0, 0, 0, 0, 0])

        np.testing.assert_array_equal(weights_truncated, expected)

    def test_truncate_weights_mixed_7_bits(self) -> None:
        """Tests truncation of weights for mixed sign mode."""
        weights_truncated = truncate_weights(
            weights=self.weights,
            sign_mode=SignMode.MIXED.value,
            num_weight_bits=7)
        expected = np.array([-256, -256, -256, -52, -4, 0, 48, 252, 252, 252])

        np.testing.assert_array_equal(weights_truncated, expected)

    def test_truncate_weights_excitatory_7_bits(self) -> None:
        """Tests truncation of weights for mixed sign mode."""
        weights_truncated = truncate_weights(
            weights=self.weights,
            sign_mode=SignMode.EXCITATORY.value,
            num_weight_bits=7)
        expected = np.array([0, 0, 0, 0, 0, 0, 48, 254, 254, 254])

        np.testing.assert_array_equal(weights_truncated, expected)

    def test_truncate_weights_inhibitory_7_bits(self) -> None:
        """Tests truncation of weights for mixed sign mode."""
        weights_truncated = truncate_weights(
            weights=self.weights,
            sign_mode=SignMode.INHIBITORY.value,
            num_weight_bits=7)
        expected = np.array([-256, -256, -256, -50, -2, 0, 0, 0, 0, 0])

        np.testing.assert_array_equal(weights_truncated, expected)
