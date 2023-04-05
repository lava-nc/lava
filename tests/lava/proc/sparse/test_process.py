# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from scipy.sparse import csr_matrix, spmatrix

from lava.proc.sparse.process import Sparse 


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


