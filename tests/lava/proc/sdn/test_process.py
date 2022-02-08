# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import numpy as np
from lava.proc.sdn.process import SigmaDelta, ACTIVATION_MODE


class TestSigmaDeltaProcess(unittest.TestCase):
    """Tests for SigmaDelta class"""
    def test_init(self) -> None:
        """Tests instantiation of SigmaDelta"""
        sdn = SigmaDelta(shape=(2, 3, 4), vth=10)

        self.assertEqual(sdn.shape, (2, 3, 4))
        self.assertEqual(sdn.vth.init, 10)
        self.assertEqual(sdn.cum_error.init, False)
        self.assertEqual(sdn.bias.init, 0)
        self.assertEqual(sdn.wgt_exp.init, 6)
        self.assertEqual(sdn.state_exp.init, 6)
        self.assertEqual(sdn.proc_params['act_fn'], ACTIVATION_MODE.ReLU)
