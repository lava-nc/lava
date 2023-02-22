# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from time import sleep

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunContinuous, RunSteps
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class SimpleProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)


@implements(proc=SimpleProcess, protocol=LoihiProtocol)
@requires(CPU)
class SimpleProcessModel(PyLoihiProcessModel):
    """
    Defines a SimpleProcessModel
    """
    u = LavaPyType(int, int)
    v = LavaPyType(int, int)


class SimpleRunConfig(RunConfig):
    """
    Defines a simple run config
    """

    def __init__(self, **kwargs):
        sync_domains = kwargs.pop("sync_domains")
        super().__init__(custom_sync_domains=sync_domains)
        self.model = None
        if "model" in kwargs:
            self.model = kwargs.pop("model")

    def select(self, process, proc_models):
        if self.model is not None:
            if self.model == "sub" and isinstance(process, SimpleProcess):
                return proc_models[1]

        return proc_models[0]


class TestRunContinuous(unittest.TestCase):
    def test_run_continuous_sync(self):
        """
        Verifies working of a Synchronous Process in Run Continuous Mode.
        """
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunContinuous(), run_cfg=run_config)
        sleep(2)
        process.stop()

    def test_run_continuous_sync_pause(self):
        """Verifies working of pause with a Synchronous Process in a
        run continuous mode."""
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunContinuous(), run_cfg=run_config)
        process.pause()
        process.run(condition=RunContinuous())
        process.stop()

    def test_run_sync_pause(self):
        """
        Verifies working of pause with a Synchronous Process.
        """
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(200), run_cfg=run_config)
        process.pause()
        process.run(condition=RunSteps(200))
        process.stop()

    def test_wait_from_runtime(self):
        """Checks non blocking mode run of a function"""

        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=10000, blocking=False),
                    run_cfg=run_config)

        process.wait()
        process.run(condition=RunSteps(num_steps=10, blocking=False),
                    run_cfg=run_config)
        process.wait()
        process.stop()


if __name__ == '__main__':
    unittest.main()
