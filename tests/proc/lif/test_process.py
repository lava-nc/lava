# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import numpy as np
from lava.proc.lif.process import LIF


class TestLIFProcess(unittest.TestCase):
    """Tests for LIF class"""
    def test_init(self):
        """Tests instantiation of LIF"""
        lif = LIF(shape=(100,),
                  du=100*np.ones((100,), dtype=np.float),
                  dv=np.ones((100,), dtype=np.float),
                  bias=2*np.ones((100,), dtype=np.float),
                  bias_exp=np.ones((100,), dtype=np.float),
                  vth=np.ones((100,), dtype=np.float))

        self.assertEqual(lif.shape, (100,))
        self.assertListEqual(lif.du.init.tolist(), 100 * [100.])
        self.assertListEqual(lif.dv.init.tolist(), 100 * [1.])
        self.assertListEqual(lif.bias.init.tolist(), 100 * [2.])
        self.assertListEqual(lif.bias_exp.init.tolist(), 100 * [1.])
        self.assertListEqual(lif.vth.init.tolist(), 100 * [1.])
