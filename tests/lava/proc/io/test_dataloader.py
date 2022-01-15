# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import List, Tuple
import unittest
import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.proc.io.sink import RingBuffer as ReceiveProcess
from lava.proc.io.dataloader import State, Spike

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig


class TestRunConfig(RunConfig):
    """Run configuration selects appropriate Conv ProcessModel based on tag:
    floating point precision or Loihi bit-accurate fixed point precision"""
    def __init__(self, select_tag: str = 'fixed_pt'):
        super().__init__(custom_sync_domains=None)
        self.select_tag = select_tag

    def select(
        self, _, proc_models: List[PyLoihiProcessModel]
    ) -> PyLoihiProcessModel:
        # print(proc_models)
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError('No legal ProcessModel found.')


class DummyDataset:
    def __init__(self, shape: tuple) -> None:
        self.shape = shape

    def __len__(self) -> int:
        return 10

    def __getitem__(self, id: int) -> Tuple[np.ndarray, int]:
        data = np.arange(np.prod(self.shape)).reshape(self.shape) + id
        data = data % np.prod(self.shape)
        label = id
        return data, label


class SpikeDataset(DummyDataset):
    def __getitem__(self, id: int) -> Tuple[np.ndarray, int]:
        data = np.arange(np.prod(self.shape)).reshape(self.shape[::-1]) + id
        data = data.transpose(np.arange(len(self.shape))[::-1]) % 13
        data = data >= 10
        label = id
        return data, label


class DummyProc(AbstractProcess):
    def __init__(self, shape: tuple) -> None:
        super().__init__(shape=shape)
        self.state = Var(shape=shape, init=np.zeros(shape))
        self.s_out = OutPort(shape=shape)


@implements(proc=DummyProc, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyDummyProc(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    state: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self) -> None:
        self.s_out.send(self.state)


class TestDataloader(unittest.TestCase):
    def test_state_loader(self) -> None:
        """Tests state dataloder"""
        num_steps = 30
        shape = (5, 7)
        interval = 5
        offset = 2

        proc = DummyProc(shape)
        dataloader = State(DummyDataset(shape), interval, offset)
        gt = ReceiveProcess(shape=dataloader.gt.shape, buffer=num_steps)
        out = ReceiveProcess(shape=shape, buffer=num_steps)
        dataloader.connect_var(proc.state)
        dataloader.gt.connect(gt.a_in)
        proc.s_out.connect(out.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        proc.run(condition=run_condition, run_cfg=run_config)
        gt_data = gt.data.get()
        out_data = out.data.get()
        proc.stop()

        dataset = DummyDataset(shape)
        for i in range(offset + 1, num_steps):
            id = (i - offset - 1) // interval
            data, gt = dataset[id]
            self.assertTrue(np.array_equal(gt, gt_data[..., i].item()))
            self.assertTrue(np.array_equal(data, out_data[..., i]))

    def test_spike_loader(self) -> None:
        """Tests spike dataloder"""
        shape = (5, 7)
        interval = 5
        offset = 2
        num_steps = 31 + offset

        dataloader = Spike(SpikeDataset(shape + (interval,)), interval, offset)
        gt = ReceiveProcess(shape=dataloader.gt.shape, buffer=num_steps)
        out = ReceiveProcess(shape=shape, buffer=num_steps)
        dataloader.gt.connect(gt.a_in)
        dataloader.s_out.connect(out.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        dataloader.run(condition=run_condition, run_cfg=run_config)
        gt_data = gt.data.get()
        out_data = out.data.get()
        dataloader.stop()

        dataset = SpikeDataset(shape + (interval,))
        for i in range(offset + 1, num_steps, interval):
            id = (i - offset - 1) // interval
            data, gt = dataset[id]
            gt_error = np.abs(gt_data[..., i:i + interval] - gt).sum()
            spike_error = np.abs(out_data[..., i:i + interval] - data).sum()
            self.assertTrue(gt_error == 0)
            self.assertTrue(spike_error == 0)

    def test_spike_loader_less_steps(self) -> None:
        """Tests spike dataloder when data load interval is less than sample
        time steps"""
        shape = (5, 7)
        steps = 3
        interval = 5
        offset = 2
        num_steps = 31 + offset

        dataloader = Spike(SpikeDataset(shape + (steps,)), interval, offset)
        gt = ReceiveProcess(shape=dataloader.gt.shape, buffer=num_steps)
        out = ReceiveProcess(shape=shape, buffer=num_steps)
        dataloader.gt.connect(gt.a_in)
        dataloader.s_out.connect(out.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        dataloader.run(condition=run_condition, run_cfg=run_config)
        gt_data = gt.data.get()
        out_data = out.data.get()
        dataloader.stop()

        dataset = SpikeDataset(shape + (steps,))
        for i in range(offset + 1, num_steps, interval):
            id = (i - offset - 1) // interval
            data, gt = dataset[id]
            gt_error = np.abs(gt_data[..., i:i + interval] - gt).sum()
            spike_error = np.abs(out_data[..., i:i + steps] - data).sum()
            gap_error = np.abs(out_data[..., i + steps:i + interval]).sum()
            self.assertTrue(gt_error == 0)
            self.assertTrue(spike_error == 0)
            self.assertTrue(gap_error == 0)

    def test_spike_loader_more_steps(self) -> None:
        """Tests spike dataloder when data load interval is less than sample
        time steps."""
        shape = (5, 7)
        steps = 8
        interval = 5
        offset = 2
        num_steps = 31 + offset

        dataloader = Spike(SpikeDataset(shape + (steps,)), interval, offset)
        gt = ReceiveProcess(shape=dataloader.gt.shape, buffer=num_steps)
        out = ReceiveProcess(shape=shape, buffer=num_steps)
        dataloader.gt.connect(gt.a_in)
        dataloader.s_out.connect(out.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        dataloader.run(condition=run_condition, run_cfg=run_config)
        gt_data = gt.data.get()
        out_data = out.data.get()
        dataloader.stop()

        dataset = SpikeDataset(shape + (steps,))
        for i in range(offset + 1, num_steps, interval):
            id = (i - offset - 1) // interval
            data, gt = dataset[id]
            gt_error = np.abs(gt_data[..., i:i + interval] - gt).sum()
            spike_error = np.abs(
                out_data[..., i:i + interval] - data[..., :interval]
            ).sum()
            self.assertTrue(gt_error == 0)
            self.assertTrue(spike_error == 0)


if __name__ == '__main__':
    num_steps = 30
    shape = (5, 7)

    dataloader = Spike(SpikeDataset(shape + (5,)), interval=5)
    gt = ReceiveProcess(shape=dataloader.gt.shape, buffer=num_steps)
    out = ReceiveProcess(shape=shape, buffer=num_steps)
    dataloader.gt.connect(gt.a_in)
    dataloader.s_out.connect(out.a_in)

    run_condition = RunSteps(num_steps=num_steps)
    run_config = TestRunConfig(select_tag='fixed_pt')
    dataloader.run(condition=run_condition, run_cfg=run_config)
    gt_data = gt.data.get()
    out_data = out.data.get()
    dataloader.stop()

    print(f'{gt_data=}')
    print(f'{out_data=}')
