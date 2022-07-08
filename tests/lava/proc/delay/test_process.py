# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.proc.delay.process import Delay


class TestConnProcess(unittest.TestCase):
    """Tests for Delay class"""

    def test_init(self):
        """Tests instantiation of Delay"""
        shape = (100, 200)
        weights = np.random.randint(100, size=shape)
        delays = np.random.randint(100, size=shape)

        conn = Delay(
            shape=shape,
            weights=weights,
            delays=delays
        )

        self.assertEqual(np.shape(conn.weights.init), shape)
        self.assertIsNone(
            np.testing.assert_array_equal(conn.weights.init, weights)
        )
        self.assertEqual(np.shape(conn.delays.init), shape)
        self.assertIsNone(
            np.testing.assert_array_equal(conn.delays.init, delays)
        )

    def test_no_in_args(self):
        """Tests instantiation of Delay with no input arguments"""
        conn = Delay()
        self.assertEqual(np.shape(conn.weights.init), (1, 1))
        self.assertEqual(np.shape(conn.delays.init), (1, 1))

    def test_input_validation_shape(self):
        """Tests input validation on the dimensions of 'shape'. (Must be 2D.)"""
        shape = (100, 200, 300)
        with self.assertRaises(AssertionError):
            Delay(shape=shape)

    def test_input_validation_weights(self):
        """Tests input validation on the dimensions of 'weights'. (Must be
        2D.)"""
        weights = np.random.randint(100, size=(2, 3, 4))
        with self.assertRaises(AssertionError):
            Delay(weights=weights)

    def test_input_validation_delays(self):
        """Tests input validation on the dimensions of 'delays'. (Must be
        2D.)"""
        delays = np.random.randint(100, size=(2, 3, 4))
        with self.assertRaises(AssertionError):
            Delay(delays=delays)
