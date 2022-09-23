# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.learning.utils import saturate, stochastic_round


class TestSaturate(unittest.TestCase):
    def test_saturate_float(self) -> None:
        """Tests saturate method with values being float."""
        min_value = 0
        values = np.array([-1, 5, 15], dtype=float)
        max_value = 10
        target_saturated_values = np.array([0, 5, 10], dtype=float)

        saturated_values = saturate(min_value, values, max_value)

        self.assertIsInstance(saturated_values, np.ndarray)
        self.assertEqual(saturated_values.dtype, float)
        np.testing.assert_almost_equal(saturated_values,
                                       target_saturated_values)

    def test_saturate_integer(self) -> None:
        """Tests saturate method with values being int."""
        min_value = 0
        values = np.array([-1, 5, 15], dtype=int)
        max_value = 10
        target_saturated_values = np.array([0, 5, 10], dtype=int)

        saturated_values = saturate(min_value, values, max_value)

        self.assertIsInstance(saturated_values, np.ndarray)
        self.assertEqual(saturated_values.dtype, int)
        np.testing.assert_almost_equal(saturated_values,
                                       target_saturated_values)


class TestStochasticRound(unittest.TestCase):
    def test_stochastic_round_float_probabilities(self) -> None:
        """Tests stochastic_round method with random_number and probabilities
        being float."""
        value = np.array([2, 5, 10, 11, 15], dtype=int)
        random_number = 0.5364
        probability = np.array([0.7459, 0.1456, 0.9149, 0.3795, 0.2976])

        target_rounded_value = np.array([3, 5, 11, 11, 15], dtype=int)

        rounded_value = stochastic_round(value, random_number, probability)

        self.assertIsInstance(rounded_value, np.ndarray)
        self.assertEqual(rounded_value.dtype, int)
        np.testing.assert_almost_equal(rounded_value,
                                       target_rounded_value)

    def test_stochastic_round_integer_probabilities(self) -> None:
        """Tests stochastic_round method with random_number and probabilities
        being int."""
        value = np.array([2, 5, 10, 11, 15], dtype=int)
        random_number = 5364
        probability = np.array([7459, 1456, 9149, 3795, 2976], dtype=int)

        target_rounded_value = np.array([3, 5, 11, 11, 15], dtype=int)

        rounded_value = stochastic_round(value, random_number, probability)

        self.assertIsInstance(rounded_value, np.ndarray)
        self.assertEqual(rounded_value.dtype, int)
        np.testing.assert_almost_equal(rounded_value,
                                       target_rounded_value)


if __name__ == "__main__":
    unittest.main()
