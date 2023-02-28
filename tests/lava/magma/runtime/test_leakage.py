# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import platform

import numpy as np
import psutil

from lava.magma.core.run_conditions import RunSteps
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF
from lava.magma.core.run_configs import Loihi1SimCfg


def run_simulation() -> None:
    lif1 = LIF(shape=(1,))
    dense = Dense(weights=np.eye(1))
    lif2 = LIF(shape=(1,))
    lif1.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(lif2.in_ports.a_in)
    lif1.run(condition=RunSteps(num_steps=10), run_cfg=Loihi1SimCfg())
    lif1.stop()


class TestLeakage(unittest.TestCase):
    @unittest.skipIf(
        platform.system() == "Windows", "Windows has no file descriptors"
    )
    def test_leakage(self):
        # initial run to make sure all components are initialized
        run_simulation()

        process = psutil.Process()
        num_fds_before = process.num_fds()

        # file descriptors opened by further runs should be closed
        run_simulation()

        num_fds_after = process.num_fds()
        self.assertEqual(num_fds_before, num_fds_after)
