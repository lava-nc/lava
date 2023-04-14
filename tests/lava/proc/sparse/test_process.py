# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from scipy.sparse import csr_matrix, spmatrix

from lava.proc.sparse.process import Sparse, LearningSparse, DelaySparse
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi


class TestSparseProcess(unittest.TestCase):
    """Tests for Sparse class"""

    def test_init(self):
        """Tests instantiation of Sparse"""
        shape = (100, 200)
        weights = np.random.random(shape)
        
        # sparsify
        weights[weights < 0.7] = 0

        # convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = Sparse(weights=weights_sparse)

        self.assertIsInstance(conn.weights.init, spmatrix)
        np.testing.assert_array_equal(conn.weights.init.toarray(), weights)


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
        
        # sparsify
        weights[weights < 0.7] = 0

        # convert to spmatrix
        weights_sparse = csr_matrix(weights)

        conn = LearningSparse(weights=weights_sparse,
                              learning_rule=learning_rule)

        self.assertIsInstance(conn.weights.init, spmatrix)
        np.testing.assert_array_equal(conn.weights.init.toarray(), weights)


class TestDelaySparseProcess(unittest.TestCase):
    """Tests for Sparse class"""

    def test_init(self):
        """Tests instantiation of Sparse"""
        shape = (100, 200)
        weights = np.random.random(shape)
        delays = np.random.randint(0,3, shape)

        # sparsify
        weights[weights < 0.7] = 0
        delays[weights < 0.7] = 0

        # convert to spmatrix
        weights_sparse = csr_matrix(weights)
        delays_sparse= csr_matrix(delays)

        conn = DelaySparse(weights=weights_sparse, delays=delays_sparse)

        self.assertIsInstance(conn.weights.init, spmatrix)
        np.testing.assert_array_equal(conn.weights.init.toarray(), weights)
