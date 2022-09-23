# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.proc.dense.process import Dense, SignMode, LearningDense


class TestDenseProcess(unittest.TestCase):
    """Tests for Dense class"""

    def test_init(self):
        """Tests instantiation of Dense"""
        shape = (100, 200)
        weights = np.random.randint(100, size=shape)

        conn = Dense(weights=weights)

        self.assertEqual(np.shape(conn.weights.init), shape)
        np.testing.assert_array_equal(conn.weights.init, weights)

    def test_input_validation_weights(self):
        """Tests input validation on the dimensions of 'weights'. (Must be
        2D.)"""
        weights = np.random.randint(100, size=(2, 3, 4))
        with self.assertRaises(ValueError):
            Dense(weights=weights)


class TestLearningDenseProcess(unittest.TestCase):
    """Tests for LearningDense class"""

    def test_init(self):
        """Tests instantiation of LearningDense"""
        shape = (100, 200)
        weights = np.random.randint(100, size=shape)

        conn = LearningDense(weights=weights)

        self.assertEqual(np.shape(conn.weights.init), shape)
        np.testing.assert_array_equal(conn.weights.init, weights)
