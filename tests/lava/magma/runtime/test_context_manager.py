# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import abc
import unittest
from typing import Optional
from time import sleep

from lava.magma.compiler.compiler import Compiler
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunContinuous, RunSteps
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.runtime.runtime import Runtime


class SimpleProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)


@implements(proc=SimpleProcess, protocol=LoihiProtocol)
@requires(CPU)
class SimpleProcessModel(PyLoihiProcessModel):
    u = LavaPyType(int, int)
    v = LavaPyType(int, int)


class SimpleRunConfig(RunConfig):
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


class Stoppable(abc.ABC):
    @abc.abstractmethod
    def stop(self) -> None:
        ...


class TestContextManager(unittest.TestCase):
    def setUp(self) -> None:
        self.stoppable: Optional[Stoppable] = None

    def tearDown(self) -> None:
        """
        Ensures process/runtime is stopped if context manager fails to.
        """
        if self.stoppable is not None:
            self.stoppable.stop()

    def test_context_manager_stops_process_continuous(self):
        """
        Verifies context manager stops process when exiting "with" block.
        Process is started with RunContinuous.
        """
        process = SimpleProcess(shape=(2, 2))
        self.stoppable = process
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])

        with process:
            process.run(condition=RunContinuous(), run_cfg=run_config)
            self.assertTrue(process.runtime._is_running)
            self.assertTrue(process.runtime._is_started)
            sleep(2)

        self.assertFalse(process.runtime._is_running)
        self.assertFalse(process.runtime._is_started)

    def test_context_manager_stops_process_runsteps_nonblocking(self):
        """
        Verifies context manager stops process when exiting "with" block.
        Process is started with RunSteps(blocking=false).
        """
        process = SimpleProcess(shape=(2, 2))
        self.stoppable = process
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])

        with process:
            process.run(condition=RunSteps(200, blocking=False),
                        run_cfg=run_config)
            self.assertTrue(process.runtime._is_running)
            self.assertTrue(process.runtime._is_started)

        self.assertFalse(process.runtime._is_running)
        self.assertFalse(process.runtime._is_started)

    def test_context_manager_stops_process_runsteps_blocking(self):
        """
        Verifies context manager stops process when exiting "with" block.
        Process is started with RunSteps(blocking=true).
        """
        process = SimpleProcess(shape=(2, 2))
        self.stoppable = process
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])

        with process:
            process.run(condition=RunSteps(200, blocking=True),
                        run_cfg=run_config)
            sleep(2)

        self.assertFalse(process.runtime._is_running)
        self.assertFalse(process.runtime._is_started)

    def test_context_manager_stops_runtime(self):
        """
        Verifies context manager stops runtime when exiting "with" block.
        """
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        compiler = Compiler()
        executable = compiler.compile(process, run_config)
        runtime = Runtime(executable,
                          ActorType.MultiProcessing)
        executable.assign_runtime_to_all_processes(runtime)

        self.stoppable = runtime

        with runtime:
            self.assertTrue(runtime._is_initialized)
            self.assertFalse(runtime._is_running)
            self.assertFalse(runtime._is_started)

            runtime.start(run_condition=RunContinuous())

            self.assertTrue(runtime._is_running)
            self.assertTrue(runtime._is_started)
            sleep(2)

        self.assertFalse(runtime._is_running)
        self.assertFalse(runtime._is_started)


if __name__ == '__main__':
    unittest.main()
