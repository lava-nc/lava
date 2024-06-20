# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from scipy.sparse import csr_matrix

import lava.frameworks.loihi2 as lv


class TestNetworks(unittest.TestCase):
    """Tests for LVA Networks."""

    def test_networks_instantiate(self):
        """Tests if LVA Networks can be instantiated."""
        inputvec = lv.InputVec(np.ones((1,)), shape=(1,))
        outputvec = lv.OutputVec(shape=(1,), buffer=1)
        threshvec = lv.GradedVec(shape=(1,))
        gradeddense = lv.GradedDense(weights=np.ones((1, 1)))
        gradedsparse = lv.GradedSparse(weights=csr_matrix(np.ones((1, 1))))
        productvec = lv.ProductVec(shape=(1,))
        lifvec = lv.LIFVec(shape=(1,))
        normnet = lv.NormalizeNet(shape=(1,))


if __name__ == '__main__':
    unittest.main()
