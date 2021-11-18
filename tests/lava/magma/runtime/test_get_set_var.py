# Copyright (C) 2021 Intel Corporation
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
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class SimpleProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs["shape"]
        self.u = Var(shape=shape, init=np.array([[7, 8], [9, 10]],
                                                dtype=np.int32))
        self.v = Var(shape=shape, init=np.array([[1, 2], [4, 5]],
                                                dtype=np.int32))


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
    v = LavaPyType(np.ndarray, np.int32, precision=32)


class TestGetSetVar(unittest.TestCase):
    def test_get_set_var_using_runtime(self):
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=10), run_cfg=run_config)

        expected_result = np.array([[7, 8], [9, 10]], dtype=np.int32)
        assert np.array_equal(process._runtime.get_var(process.u.id),
                              expected_result)
        expected_result *= 10
        process._runtime.set_var(process.u.id, expected_result)
        assert np.array_equal(process._runtime.get_var(process.u.id),
                              expected_result)
        process.run(condition=RunSteps(num_steps=5), run_cfg=run_config)
        assert np.array_equal(process._runtime.get_var(process.u.id),
                              expected_result)
        self.assertEqual(process.runtime.global_time, 15)
        process.stop()

    def test_get_set_var_using_var_api(self):
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=10), run_cfg=run_config)

        expected_result = np.array([[7, 8], [9, 10]], dtype=np.int32)
        assert np.array_equal(process.u.get(),
                              expected_result)
        expected_result *= 10
        process.u.set(expected_result)
        assert np.array_equal(process.u.get(),
                              expected_result)
        process.run(condition=RunSteps(num_steps=5), run_cfg=run_config)
        assert np.array_equal(process.u.get(),
                              expected_result)
        self.assertEqual(process.runtime.global_time, 15)
        process.stop()


if __name__ == '__main__':
    unittest.main()
