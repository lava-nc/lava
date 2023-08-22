# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from scipy.sparse import csr_matrix, spmatrix

from lava.utils.sparse import find
from lava.proc.sparse.process import Sparse, LearningSparse, DelaySparse
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi


class TestFunctions(unittest.TestCase):
    """Test helper function for Sparse"""

    def test_find_with_explicit_zeros(self):
        mat = np.random.randint(-10, 10, (3, 5))
        spmat = csr_matrix(mat)
        spmat.data[0] = 0

        _, _, vals = find(spmat, explicit_zeros=True)

        self.assertTrue(np.all(spmat.data in vals))


class TestSparseProcess(unittest.TestCase):
    """Tests for Sparse class"""

    def test_init(self):
        """Tests instantiation of Sparse"""
        shape = (100, 200)
        weights = np.random.random(shape)

        # Sparsify
        weights[weights < 0.7] = 0

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = Sparse(weights=weights_sparse)

        np.testing.assert_array_equal(conn.weights.init.toarray(), weights)

    def test_init_of_sparse_with_ndarray(self):
        """Tests instantiation of Sparse with ndarray as
         weights"""

        shape = (3, 2)
        weights = np.random.random(shape)

        conn = Sparse(weights=weights)

        np.testing.assert_array_equal(conn.weights.get().toarray(), weights)


class TestLearningSparseProcess(unittest.TestCase):
    """Tests for LearningSparse class"""

    def test_init(self):
        """Tests instantiation of LearningSparse"""

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=1,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
        )

        shape = (100, 200)
        weights = np.random.random(shape)

        # Sparsify
        weights[weights < 0.7] = 0

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = LearningSparse(weights=weights_sparse,
                              learning_rule=learning_rule)

        np.testing.assert_array_equal(conn.weights.init.toarray(), weights)

    def test_init_of_learningsparse_with_ndarray(self):
        """Tests instantiation of LearningSparse with
        ndarray as weights"""

        shape = (3, 2)
        weights = np.random.random(shape)

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=1,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
        )

        conn = LearningSparse(weights=weights, learning_rule=learning_rule)

        np.testing.assert_array_equal(conn.weights.get().toarray(), weights)


class TestDelaySparseProcess(unittest.TestCase):
    """Tests for Sparse class"""

    def test_init(self):
        """Tests instantiation of Sparse"""
        shape = (100, 200)
        weights = np.random.random(shape)
        delays = np.random.randint(0, 3, shape)

        # Sparsify
        weights[weights < 0.7] = 0
        delays[weights < 0.7] = 0

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)
        delays_sparse = csr_matrix(delays)

        conn = DelaySparse(weights=weights_sparse, delays=delays_sparse)

        np.testing.assert_array_equal(conn.weights.init.toarray(), weights)

    def test_validate_shapes(self):
        """Tests if the weights and delay have correct shape"""

        shape = (3, 2)
        weights = np.random.random(shape)

        shape = (2, 3)
        delays = np.random.randint(0, 3, shape)

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)
        delays_sparse = csr_matrix(delays)

        np.testing.assert_raises(ValueError,
                                 DelaySparse,
                                 weights=weights_sparse,
                                 delays=delays_sparse)

    def test_validate_nonzero_delays(self):
        """Tests if the weights and delay have correct shape"""

        shape = (3, 2)
        weights = np.random.random(shape)
        delays = np.random.randint(0, 3, shape)
        delays[0] = -1

        # Convert to spmatrix
        weights_sparse = csr_matrix(weights)
        delays_sparse = csr_matrix(delays)

        np.testing.assert_raises(ValueError,
                                 DelaySparse,
                                 weights=weights_sparse,
                                 delays=delays_sparse)

    def test_init_of_delaysparse_with_ndarray(self):
        """Tests instantiation of DelaySparse with ndarray as weights"""

        shape = (3, 2)
        weights = np.random.random(shape)
        delays = np.random.randint(0, 3, shape)

        conn = DelaySparse(weights=weights, delays=delays)

        np.testing.assert_array_equal(conn.weights.get().toarray(), weights)


if __name__ == '__main__':
    unittest.main()
