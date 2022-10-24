# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.proc.sparse.process import Sparse


class TestSparseProcess(unittest.TestCase):
    """Tests for Sparse class"""

    def test_init(self):
        """Tests instantiation of Sparse"""
        shape = (100, 200)
        weights = np.random.randint(100, size=shape)

        conn = Sparse(weights=weights)

        self.assertEqual(np.shape(conn.weights.init), shape)
        np.testing.assert_array_equal(conn.weights.init, weights)

    def test_input_validation_weights(self):
        """Tests input validation on the dimensions of 'weights'. (Must be
        2D.)"""
        weights = np.random.randint(100, size=(2, 3, 4))
        with self.assertRaises(ValueError):
            Sparse(weights=weights)


if __name__ == "__main__":
    unittest.main()
