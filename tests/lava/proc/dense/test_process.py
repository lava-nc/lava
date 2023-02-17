# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.proc.dense.process import Dense, LearningDense
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi


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

        lr = STDPLoihi(learning_rate=0.1,
                       A_plus=1.,
                       A_minus=1.,
                       tau_plus=20.,
                       tau_minus=20.)

        conn = LearningDense(weights=weights,
                             learning_rule=lr)

        self.assertEqual(np.shape(conn.weights.init), shape)
        np.testing.assert_array_equal(conn.weights.init, weights)
