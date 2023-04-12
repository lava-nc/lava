# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class SimpleProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.u = Var(shape=shape, init=np.array([[7, 8], [9, 10]],
                                                dtype=np.int32))
        self.v = Var(shape=shape, init=np.array([[1., 2.55], [4.2, 5.1]],
                                                dtype=np.float64))


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


@implements(proc=SimpleProcess, protocol=LoihiProtocol)
@requires(CPU)
class SimpleProcessModel(PyLoihiProcessModel):
    u = LavaPyType(np.ndarray, np.int32, precision=32)
    v = LavaPyType(np.ndarray, np.float64, precision=32)


class TestGetSetVar(unittest.TestCase):
    def test_get_set_var_using_runtime(self):
        """Checks that get_var() method of the runtime retrieves expected
        values, set_var() method modifies the values and retrieve them again.
        Different data types are tested also (int and float)."""
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=10), run_cfg=run_config)

        # Retrieve value of u
        expected_result_u = np.array([[7, 8], [9, 10]], dtype=np.int32)
        self.assertTrue(np.array_equal(process._runtime.get_var(process.u.id),
                                       expected_result_u))

        # Retrieve value of v
        expected_result_v = np.array([[1., 2.55], [4.2, 5.1]], dtype=np.float64)
        self.assertTrue(np.array_equal(process._runtime.get_var(process.v.id),
                                       expected_result_v))

        # Modify value of u
        expected_result_u *= 10
        process._runtime.set_var(process.u.id, expected_result_u)

        # Check if value was modified by retrieving it again
        self.assertTrue(np.array_equal(process._runtime.get_var(process.u.id),
                                       expected_result_u))

        # Modify value of v
        expected_result_v *= 10
        process._runtime.set_var(process.v.id, expected_result_v)

        # Check if value was modified by retrieving it again
        self.assertTrue(np.array_equal(process._runtime.get_var(process.v.id),
                                       expected_result_v))

        # Check if values stay modified after another execution
        process.run(condition=RunSteps(num_steps=5), run_cfg=run_config)
        self.assertTrue(np.array_equal(process._runtime.get_var(process.u.id),
                                       expected_result_u))
        self.assertTrue(np.array_equal(process._runtime.get_var(process.v.id),
                                       expected_result_v))
        process.stop()

    def test_get_set_var_using_var_api(self):
        """Checks that get_var() method of Var retrieves expected values,
        set_var() method modifies the values and retrieve them again."""
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=10), run_cfg=run_config)

        # Retrieve value of u
        expected_result_u = np.array([[7, 8], [9, 10]], dtype=np.int32)
        self.assertTrue(np.array_equal(process.u.get(), expected_result_u))

        # Retrieve value of v
        expected_result_v = np.array([[1., 2.55], [4.2, 5.1]], dtype=np.float64)
        self.assertTrue(np.array_equal(process.v.get(), expected_result_v))

        # Modify value of u
        expected_result_u *= 10
        process.u.set(expected_result_u)

        # Check if value was modified by retrieving it again
        self.assertTrue(np.array_equal(process.u.get(), expected_result_u))

        # Modify value of v
        expected_result_v *= 10
        process.v.set(expected_result_v)

        # Check if value was modified by retrieving it again
        self.assertTrue(np.array_equal(process.v.get(), expected_result_v))

        # Check if values stay modified after another execution
        process.run(condition=RunSteps(num_steps=5), run_cfg=run_config)
        self.assertTrue(np.array_equal(process.u.get(), expected_result_u))
        self.assertTrue(np.array_equal(process.v.get(), expected_result_v))
        process.stop()

    def test_get_set_variable_set_before_next_run(self):
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=10), run_cfg=run_config)

        # Retrieve value of u
        expected_result_u = np.array([[7, 8], [9, 10]], dtype=np.int32)
        process.u.set(expected_result_u)
        # Check if values stay modified after another execution
        process.run(condition=RunSteps(num_steps=1), run_cfg=run_config)
        self.assertTrue(np.array_equal(process.u.get(), expected_result_u))
        process.stop()

    def test_get_set_variable_continous_mode(self):
        """Checks if set() and get() works when running continuously. They will
         only work when execution is paused."""

        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunContinuous(), run_cfg=run_config)

        expected_result_u = np.array([[7, 8], [9, 10]], dtype=np.int32)
        # Pause the execution
        process.pause()
        # Retrieve value of u
        self.assertTrue(np.array_equal(process.u.get(), expected_result_u))

        expected_result_u = np.array([[1, 2], [3, 4]], dtype=np.int32)
        # Set value of u
        process.u.set(expected_result_u)
        # Check if values stay modified after another execution
        process.run(condition=RunContinuous(), run_cfg=run_config)
        process.pause()
        self.assertTrue(np.array_equal(process.u.get(), expected_result_u))
        process.stop()


if __name__ == '__main__':
    unittest.main()
