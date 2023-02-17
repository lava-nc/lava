# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.proc.dense.process import Dense, LearningDense, DelayDense
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


class TestDelayDenseProcess(unittest.TestCase):
    """Tests for DelayDense class"""

    def test_init(self):
        """Tests instantiation of DelayDense"""
        shape = (100, 200)
        weights = np.random.randint(100, size=shape)
        delays = np.random.randint(10, size=shape)

        conn = DelayDense(weights=weights, delays=delays)

        self.assertEqual(np.shape(conn.weights.init), shape)
        np.testing.assert_array_equal(conn.weights.init, weights)
        np.testing.assert_array_equal(conn.delays.init, delays)

    def test_init_max_delay(self):
        """Tests that the parameter 'max_delay' creates an appropriate buffer
        'a_buff'. If 'max_delay'=15 and 'delays'=5, the dimension of a_buff
        should be [: 15+1].
        """
        shape = (100, 200)
        weights = np.random.randint(100, size=shape)
        delays = 5
        max_delay = 15
        expected_a_buff_shape = (shape[0], max_delay + 1)

        conn = DelayDense(weights=weights, delays=delays, max_delay=max_delay)

        self.assertEqual(np.shape(conn.weights.init), shape)
        np.testing.assert_array_equal(conn.weights.init, weights)
        np.testing.assert_array_equal(conn.delays.init, delays)
        np.testing.assert_array_equal(conn.a_buff.shape, expected_a_buff_shape)

    def test_input_validation_delays(self):
        """Tests input validation on the dimensions and values of 'delays'.
        (Must be 2D and positive values.)"""
        weights = np.random.randint(100, size=(2, 4))
        delays = np.random.randint(10, size=(3, 4))

        with self.assertRaises(ValueError):
            DelayDense(weights=weights, delays=delays)
        delays = -1
        with self.assertRaises(ValueError):
            DelayDense(weights=weights, delays=delays)
        delays = 1.2
        with self.assertRaises(ValueError):
            DelayDense(weights=weights, delays=delays)
        delays = np.random.rand(3, 4)
        with self.assertRaises(ValueError):
            DelayDense(weights=weights, delays=delays)
