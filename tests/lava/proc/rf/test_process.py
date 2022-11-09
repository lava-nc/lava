# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.proc.rf.process import RF


class TestRFProcess(unittest.TestCase):
    """Tests for rf class"""
    def test_init(self):
        """Tests instantiation of RF"""
        rf = RF(shape=(100,),
                period=100,
                alpha=.1,
                vth=1,
                name="RF")

        sin_decay = (1 - .1) * np.sin(np.pi * 2 * 1 / 100)
        cos_decay = (1 - .1) * np.cos(np.pi * 2 * 1 / 100)

        self.assertEqual(rf.shape, (100,))
        self.assertEqual(rf.name, "RF")
        self.assertEqual(rf.sin_decay.init, sin_decay)
        self.assertEqual(rf.cos_decay.init, cos_decay)
        self.assertEqual(rf.vth.init, 1.)
        self.assertEqual(rf.proc_params["shape"], (100,))
