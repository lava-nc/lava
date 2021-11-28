# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.proc.dense.process import Dense


class TestConnProcess(unittest.TestCase):
    """Tests for Dense class"""

    def test_init(self):
        """Tests instantiation of Dense"""
        shape = (100, 200)
        weights = np.random.randint(100, size=shape)
        weight_exp = 2
        num_weight_bits = 7
        sign_mode = 1

        conn = Dense(shape=shape, weights=weights, weight_exp=weight_exp,
                     num_weight_bits=num_weight_bits, sign_mode=sign_mode)

        self.assertEqual(np.shape(conn.weights.init), shape)
        self.assertIsNone(
            np.testing.assert_array_equal(conn.weights.init, weights))
        self.assertEqual(conn.weight_exp.init, weight_exp)
        self.assertEqual(conn.num_weight_bits.init, num_weight_bits)
        self.assertEqual(conn.sign_mode.init, sign_mode)

    def test_input_validation_shape(self):
        """Tests input validation on the dimensions of 'shape'. (Must be 2D.)"""
        shape = (100, 200, 300)
        with self.assertRaises(AssertionError):
            Dense(shape=shape)

    def test_input_validation_weights(self):
        """Tests input validation on the dimensions of 'weights'. (Must be
        2D.)"""
        weights = np.random.randint(100, size=(2, 3, 4))
        with self.assertRaises(AssertionError):
            Dense(weights=weights)
