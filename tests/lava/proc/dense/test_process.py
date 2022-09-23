# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.proc.dense.process import Dense, SignMode, LearningDense


class TestConnProcess(unittest.TestCase):
    """Tests for Dense class"""

    def test_init(self):
        """Tests instantiation of Dense"""
        shape = (100, 200)
        weights = np.random.randint(100, size=shape)
        weight_exp = 2
        num_weight_bits = 7
        sign_mode = SignMode.MIXED

        conn = Dense(
            weights=weights,
            weight_exp=weight_exp,
            num_weight_bits=num_weight_bits,
            sign_mode=sign_mode,
        )

        self.assertEqual(np.shape(conn.weights.init), shape)
        self.assertIsNone(
            np.testing.assert_array_equal(conn.weights.init, weights)
        )
        self.assertEqual(conn.weight_exp.init, weight_exp)
        self.assertEqual(conn.num_weight_bits.init, num_weight_bits)
        self.assertEqual(conn.sign_mode.init, sign_mode.value)

    def test_input_validation_weights(self):
        """Tests input validation on the dimensions of 'weights'. (Must be
        2D.)"""
        weights = np.random.randint(100, size=(2, 3, 4))
        with self.assertRaises(ValueError):
            Dense(weights=weights)


class TestPlasticConnProcess(unittest.TestCase):
    """Tests for Dense class"""

    def test_init(self):
        """Tests instantiation of Dense"""
        shape = (100, 200)
        weights = np.random.randint(100, size=shape)
        weight_exp = 2
        num_weight_bits = 7
        sign_mode = SignMode.MIXED

        conn = LearningDense(
            weights=weights,
            weight_exp=weight_exp,
            num_weight_bits=num_weight_bits,
            sign_mode=sign_mode,
        )

        self.assertEqual(np.shape(conn.weights.init), shape)
        self.assertIsNone(
            np.testing.assert_array_equal(conn.weights.init, weights)
        )
        self.assertEqual(conn.weight_exp.init, weight_exp)
        self.assertEqual(conn.num_weight_bits.init, num_weight_bits)
        self.assertEqual(conn.sign_mode.init, sign_mode.value)
