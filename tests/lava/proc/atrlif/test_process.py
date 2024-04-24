# Copyright (C) 2024 Intel Corporation
# Copyright (C) 2024 Jannik Luboeinski
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.proc.atrlif.process import ATRLIF


class TestATRLIFProcess(unittest.TestCase):
    """Tests for ATRLIF class"""
    def test_init(self):
        """Tests instantiation of ATRLIF neuron"""
        N = 100
        delta_i = 0.6
        delta_v = 0.6
        delta_theta = 0.4
        delta_r = 0.4
        theta_0 = 4
        theta_step = 2
        bias_mant = 2 * np.ones((N,), dtype=float)
        bias_exp = np.ones((N,), dtype=float)
        name = "ATRLIF"

        neur = ATRLIF(shape=(N,),
                      delta_i=delta_i,
                      delta_v=delta_v,
                      delta_theta=delta_theta,
                      delta_r=delta_r,
                      theta_0=theta_0,
                      theta=theta_0,
                      theta_step=theta_step,
                      bias_mant=bias_mant,
                      bias_exp=bias_exp,
                      name=name)

        self.assertEqual(neur.proc_params["shape"], (N,))
        self.assertEqual(neur.delta_i.init, delta_i)
        self.assertEqual(neur.delta_v.init, delta_v)
        self.assertListEqual(neur.bias_mant.init.tolist(), bias_mant.tolist())
        self.assertListEqual(neur.bias_exp.init.tolist(), bias_exp.tolist())
        self.assertEqual(neur.name, name)
