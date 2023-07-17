# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyAsyncProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunContinuous, RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol


class AsyncProcess1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)


class AsyncProcess2(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)


@implements(proc=AsyncProcess1, protocol=AsyncProtocol)
@requires(CPU)
class AsyncProcessModel1(PyAsyncProcessModel):
    u = LavaPyType(np.ndarray, np.int32)
    v = LavaPyType(np.ndarray, np.int32)

    def run_async(self):
        while True:
            self.u = self.u + 10
            self.v = self.v + 1000
            if self.check_for_stop_cmd():
                return
            if self.check_for_pause_cmd():
                return


@implements(proc=AsyncProcess2, protocol=AsyncProtocol)
@requires(CPU)
class AsyncProcessModel2(PyAsyncProcessModel):
    u = LavaPyType(np.ndarray, np.int32)
    v = LavaPyType(np.ndarray, np.int32)
    steps = 0

    def run_async(self):
        while self.steps < self.num_steps:
            self.u = self.u + 10
            self.v = self.v + 1000
            self.steps += 1


class TestProcess(unittest.TestCase):
    def test_async_process_model(self):
        """
        Verifies the working of Asynchronous Process
        """
        process = AsyncProcess1(shape=(2, 2))
        _ = SyncDomain("simple", AsyncProtocol(), [process])
        process.run(condition=RunContinuous(), run_cfg=Loihi2SimCfg())
        process.stop()

    def test_async_process_model_pause(self):
        """
        Verifies the working of Asynchronous Process, pause should have
        effect
        """
        process = AsyncProcess1(shape=(2, 2))
        _ = SyncDomain("simple", AsyncProtocol(), [process])
        process.run(condition=RunContinuous(), run_cfg=Loihi2SimCfg())
        process.pause()
        process.stop()

    def test_async_process_num_steps(self):
        """
        Verifies the working of Asynchronous Process, numsteps should be
        implicitly passed as num_steps for the process.
        """
        process = AsyncProcess2(shape=(2, 2))
        _ = SyncDomain("simple", AsyncProtocol(), [process])
        process.run(condition=RunSteps(num_steps=10), run_cfg=Loihi2SimCfg())
        process.stop()

    def test_async_process_get(self):
        """
        Verifies the working of Asynchronous Process, get should get the value
        of the variable after run finishes.
        """
        process = AsyncProcess2(shape=(2, 2))
        _ = SyncDomain("simple", AsyncProtocol(), [process])
        process.run(condition=RunSteps(num_steps=10), run_cfg=Loihi2SimCfg())
        print(process.u.get())
        process.stop()


if __name__ == "__main__":
    unittest.main()
