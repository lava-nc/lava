# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import asteval
import numpy as np

from lava.magma.core.learning.utils import stochastic_round, float_to_literal


class TestUtils(unittest.TestCase):
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

    def test_float_to_literals(self) -> None:
        """Tests if float_to_literal works correctly."""
        aeval = asteval.Interpreter()

        def reconstruct(x):
            literal = float_to_literal(x)
            return aeval(literal.replace("^", "**").replace("2", "2."))

        values = np.array([0, 1, 2, 3, 0.5, 0.25, 0.125])

        for v in values:
            np.testing.assert_almost_equal(reconstruct(v), v)
            np.testing.assert_almost_equal(reconstruct(-v), -v)


if __name__ == "__main__":
    unittest.main()
