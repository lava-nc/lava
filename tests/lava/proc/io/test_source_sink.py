# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from typing import List
import unittest
import numpy as np
from lava.magma.core.model.py.ports import PyOutPort

from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.source import RingBuffer as SendProcess
from lava.proc.io.sink import RingBuffer as ReceiveProcess
from lava.proc.io.sink import ReadVar
from lava.proc.io.reset import ResetVar

from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType

np.random.seed(7739)


# simple integrator class to interact with read and reset processes
class Integrator(AbstractProcess):
    def __init__(self, delta: int) -> None:
        super().__init__()
        self.shape = (1,)
        self.state = Var(self.shape, 0)
        self.delta = Var(self.shape, delta)
        self.out = OutPort(self.shape)


@implements(proc=Integrator, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyIntegrator(PyLoihiProcessModel):
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    state: np.ndarray = LavaPyType(np.ndarray, int)
    delta: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self) -> None:
        self.state += self.delta
        self.out.send(self.state)


class TestRunConfig(RunConfig):
    """Run configuration selects appropriate ProcessModel based on tag
    """
    def __init__(self, select_tag: str = 'fixed_pt'):
        super(TestRunConfig, self).__init__(custom_sync_domains=None)
        self.select_tag = select_tag

    def select(
        self, _, proc_models: List[PyLoihiProcessModel]
    ) -> PyLoihiProcessModel:
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError('No legal ProcessModel found.')


class TestSendReceive(unittest.TestCase):
    """Tests for all SendProces and ReceiveProcess."""

    def test_source_sink(self) -> None:
        """Test whatever is being sent form source is received at sink."""
        num_steps = 10
        shape = (64, 64, 16)
        input = np.random.randint(256, size=shape + (num_steps,))
        input -= 128
        # input = 0.5 * input

        source = SendProcess(data=input)
        sink = ReceiveProcess(shape=shape, buffer=num_steps)
        source.out_ports.s_out.connect(sink.in_ports.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='floating_pt')
        sink.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        sink.stop()

        self.assertTrue(
            np.all(output == input),
            f'Input and Ouptut do not match.\n'
            f'{output[output!=input]=}\n'
            f'{input[output!=input] =}\n'
        )

    def test_read_var(self) -> None:
        num_steps = 15
        delta = 5
        interval = 4
        offset = 2
        integrator = Integrator(delta)
        logger = ReadVar(
            integrator.state,
            num_steps // interval + 1, interval, offset
        )

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        integrator.run(condition=run_condition, run_cfg=run_config)
        output = logger.data.get()
        integrator.stop()

        # + delta because we are reading data after state is changed
        ground_truth = np.arange(num_steps) * delta + delta
        ground_truth = ground_truth[offset::interval]
        ground_truth.reshape(1, -1)

        error = np.abs(output[..., :len(ground_truth)] - ground_truth).sum()

        self.assertTrue(
            error == 0,
            f'Read Var has errors. Expected {ground_truth=}, found {output=}.'
        )

    def test_reset_var(self) -> None:
        num_steps = 15
        delta = 5
        interval = 4
        offset = 2
        integrator = Integrator(delta)
        # TODO: DISCUSS
        # It is not possible to attach two RefPort to same var
        # so ReadVar and ResetVar cannot be used on same process
        # logger = ReadVar(
        #     integrator.state,
        #     num_steps // interval + 1, interval, offset + 1
        # )
        ResetVar(integrator.state, -delta, interval, offset)
        sink = ReceiveProcess(shape=integrator.shape, buffer=num_steps)
        integrator.out.connect(sink.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        integrator.run(condition=run_condition, run_cfg=run_config)
        # output = logger.data.get()
        output = sink.data.get()
        integrator.stop()

        # + delta because we are reading data after state is changed
        ground_truth = np.arange(num_steps) + 1
        ground_truth[offset:] -= ground_truth[offset]
        ground_truth = (ground_truth % interval) * delta
        error = np.abs(output - ground_truth).sum()

        self.assertTrue(
            error == 0,
            f'Read Var has errors. Expected {ground_truth=}, found {output=}.'
        )
