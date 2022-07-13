# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from time import time, sleep

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel, \
    PyAsyncProcessModel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class P1(AbstractProcess):
    pass


class P2(AbstractProcess):
    pass


class P3(AbstractProcess):
    pass


class P4(AbstractProcess):
    pass


class P5(AbstractProcess):
    pass


class P6(AbstractProcess):
    pass


class SimpleRunConfig(RunConfig):
    def __init__(self, **kwargs):
        sync_domains = kwargs.pop("sync_domains")
        super().__init__(custom_sync_domains=sync_domains)
        self.model = None
        if "model" in kwargs:
            self.model = kwargs.pop("model")

    def select(self, process, proc_models):
        return proc_models[0]


@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
class P1Model(PyLoihiProcessModel):
    def run_spk(self):
        if self.time_step == 3:
            self._req_pause = True
        else:
            sleep(1)


@implements(proc=P2, protocol=LoihiProtocol)
@requires(CPU)
class P2Model(PyLoihiProcessModel):
    def pre_guard(self):
        return True

    def run_pre_mgmt(self):
        if self.time_step == 3:
            self._req_pause = True
        else:
            sleep(1)


@implements(proc=P3, protocol=LoihiProtocol)
@requires(CPU)
class P3Model(PyLoihiProcessModel):
    def lrn_guard(self):
        return True

    def run_lrn(self):
        if self.time_step == 3:
            self._req_pause = True
        else:
            sleep(1)


@implements(proc=P4, protocol=LoihiProtocol)
@requires(CPU)
class P4Model(PyLoihiProcessModel):
    def post_guard(self):
        return True

    def run_post_mgmt(self):
        if self.time_step == 3:
            self._req_pause = True
        else:
            sleep(1)


@implements(proc=P5, protocol=LoihiProtocol)
@requires(CPU)
class P5Model(PyLoihiProcessModel):
    def post_guard(self):
        return True

    def run_post_mgmt(self):
        if self.time_step == 3:
            self._req_stop = True
        else:
            sleep(1)


@implements(proc=P6, protocol=AsyncProtocol)
@requires(CPU)
class P6Model(PyAsyncProcessModel):
    def run(self):
        pass

    def async_fun(self):
        while True:
            if self.check_for_stop_cmd():
                return


class TestPauseRequestedFromModel(unittest.TestCase):
    def test_pause_request_from_model_in_spk_phase(self):
        """Ensure pause is serviced correctly when requested from the run
        function of a model"""
        s = time()
        process = P1()
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=100), run_cfg=run_config)
        e = time()
        self.assertTrue(e - s < 100, "")
        self.assertFalse(process.runtime._is_running)
        self.assertTrue(process.runtime._is_started)
        process.stop()

    def test_pause_request_from_model_in_pre_mgmt_phase(self):
        """Ensure pause is serviced correctly when requested from the run
        function of a model"""
        s = time()
        process = P2()
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=100), run_cfg=run_config)
        e = time()
        self.assertTrue(e - s < 100, "")
        self.assertFalse(process.runtime._is_running)
        self.assertTrue(process.runtime._is_started)
        process.stop()

    def test_pause_request_from_model_in_lrn_phase(self):
        """Ensure pause is serviced correctly when requested from the run
        function of a model"""
        s = time()
        process = P3()
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=100), run_cfg=run_config)
        e = time()
        self.assertTrue(e - s < 100, "")
        self.assertFalse(process.runtime._is_running)
        self.assertTrue(process.runtime._is_started)
        process.stop()

    def test_pause_request_from_model_in_post_mgmt_phase(self):
        """Ensure pause is serviced correctly when requested from the run
        function of a model"""
        s = time()
        process = P4()
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=100), run_cfg=run_config)
        e = time()
        self.assertTrue(e - s < 100, "")
        self.assertFalse(process.runtime._is_running)
        self.assertTrue(process.runtime._is_started)
        process.stop()

    def test_stop_request_from_model_in_post_mgmt_phase(self):
        """Ensure pause is serviced correctly when requested from the run
        function of a model"""
        s = time()
        process = P5()
        simple_sync_domain = SyncDomain("simple", LoihiProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunSteps(num_steps=100), run_cfg=run_config)
        e = time()
        self.assertTrue(e - s < 100, "")
        self.assertFalse(process.runtime._is_running)

    @unittest.skip
    def test_pause_request_from_hierarchical_model(self):
        """Test 2 models - one sync and another aynsc - able to pause and
        stop"""
        s = time()
        p4 = P4()
        p6 = P6()
        sync_domain = SyncDomain("simple", LoihiProtocol(), [p4])
        async_domain = SyncDomain("asimple", AsyncProtocol(), [p6])
        run_config = SimpleRunConfig(sync_domains=[sync_domain, async_domain])
        p4.run(condition=RunSteps(num_steps=100), run_cfg=run_config)
        e = time()
        self.assertTrue(e - s < 100, "")
        self.assertFalse(p4.runtime._is_running)
        self.assertTrue(p4.runtime._is_started)
        p4.stop()


if __name__ == '__main__':
    unittest.main()
