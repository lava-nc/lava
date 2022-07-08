# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.proc.lif.process import LIF, TernaryLIF


class TestLIFProcess(unittest.TestCase):
    """Tests for LIF class"""
    def test_init(self):
        """Tests instantiation of LIF"""
        lif = LIF(shape=(100,),
                  du=100.,
                  dv=1.,
                  bias_mant=2 * np.ones((100,), dtype=float),
                  bias_exp=np.ones((100,), dtype=float),
                  vth=1.,
                  name="LIF")

        self.assertEqual(lif.name, "LIF")
        self.assertEqual(lif.du.init, 100.)
        self.assertEqual(lif.dv.init, 1.)
        self.assertListEqual(lif.bias_mant.init.tolist(), 100 * [2.])
        self.assertListEqual(lif.bias_exp.init.tolist(), 100 * [1.])
        self.assertEqual(lif.vth.init, 1.)
        self.assertEqual(lif.proc_params["shape"], (100,))


class TestTernaryLIFProcess(unittest.TestCase):
    """Tests for TernaryLIF class"""
    def test_init(self):
        """Tests for instantiation of Ternary LIF"""
        tlif = TernaryLIF(shape=(100,),
                          du=100.,
                          dv=1.,
                          bias_mant=2 * np.ones((100,), dtype=float),
                          bias_exp=np.ones((100,), dtype=float),
                          vth_lo=-3., vth_hi=5.)

        self.assertEqual(tlif.du.init, 100.)
        self.assertEqual(tlif.dv.init, 1.)
        self.assertListEqual(tlif.bias_mant.init.tolist(), 100 * [2.])
        self.assertListEqual(tlif.bias_exp.init.tolist(), 100 * [1.])
        self.assertEqual(tlif.vth_lo.init, -3.)
        self.assertEqual(tlif.vth_hi.init, 5.)
        self.assertEqual(tlif.proc_params["shape"], (100,))

    def test_vth_hi_lo_order(self):
        """Test if the check to assert the order of upper and lower
        thresholds works properly, i.e., we should get a ValueError if the
        lower threshold is greater than upper threshold."""

        with(self.assertRaises(ValueError)):
            _ = TernaryLIF(shape=(100,),
                           vth_lo=15.,
                           vth_hi=5.)
