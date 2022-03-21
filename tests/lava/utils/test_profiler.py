# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
import unittest

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_configs import RunConfig


# A minimal process
from lava.utils.profiler import LavaProfiler


class P(AbstractProcess):
    ...


# A minimal PyProcModel implementing P
@implements(proc=P, protocol=LoihiProtocol)
@requires(CPU)
class PyProcModel(PyLoihiProcessModel):

    def run_spk(self):
        print("Test")


# A simple RunConfig selecting always the first found process model
class MyRunCfg(RunConfig):
    def select(self, proc, proc_models):
        return proc_models[0]


class TestLavaProfiler(unittest.TestCase):
    def test_init(self):
        """TBD"""
        proc = P()
        profiler = LavaProfiler()

        self.assertTrue(isinstance(profiler, LavaProfiler))

    def test_get_energy(self):
        """TBD"""

        proc = P()
        profiler = LavaProfiler()

        # The process proc and connected processes should be profiled
        profiler.profile(proc)

        # No connections are made

        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [proc])

        # The process should compile and run without error (not doing anything)
        proc.run(RunSteps(num_steps=3, blocking=True),
                 MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        proc.stop()

        energy = profiler.get_energy()

        self.assertTrue(energy == 0)


if __name__ == '__main__':
    unittest.main()
