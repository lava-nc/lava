# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from lava.proc.sdn.process import Sigma, Delta, SigmaDelta, ActivationMode


class TestSigmaProcess(unittest.TestCase):
    """Tests for Sigma class"""
    def test_init(self) -> None:
        """Tests instantiation of Sigma"""
        sdn = Sigma(shape=(2, 3, 4))
        self.assertEqual(sdn.shape, (2, 3, 4))


class TestDeltaProcess(unittest.TestCase):
    """Tests for Delta class"""
    def test_init(self) -> None:
        """Tests instantiation of Delta"""
        sdn = Delta(shape=(2, 3, 4), vth=10)

        self.assertEqual(sdn.shape, (2, 3, 4))
        self.assertEqual(sdn.vth.init, 10)
        self.assertEqual(sdn.cum_error.init, False)
        self.assertEqual(sdn.spike_exp.init, 0)
        self.assertEqual(sdn.state_exp.init, 0)


class TestSigmaDeltaProcess(unittest.TestCase):
    """Tests for SigmaDelta class"""
    def test_init(self) -> None:
        """Tests instantiation of SigmaDelta"""
        sdn = SigmaDelta(shape=(2, 3, 4), vth=10)

        self.assertEqual(sdn.shape, (2, 3, 4))
        self.assertEqual(sdn.vth.init, 10)
        self.assertEqual(sdn.cum_error.init, False)
        self.assertEqual(sdn.bias.init, 0)
        self.assertEqual(sdn.spike_exp.init, 0)
        self.assertEqual(sdn.state_exp.init, 0)
        self.assertEqual(sdn.proc_params['act_mode'], ActivationMode.RELU)
