# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import List, Tuple
import unittest
import numpy as np
from lava.magma.core.model.model import AbstractProcessModel

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig

from lava.proc.io.sink import RingBuffer as ReceiveProcess
from lava.proc.io.dataloader import StateDataloader, SpikeDataloader


class TestRunConfig(RunConfig):
    """Run configuration selects appropriate ProcessModel based on tag:
    floating point precision or Loihi bit-accurate fixed point precision"""
    def __init__(self, select_tag: str = 'fixed_pt') -> None:
        super().__init__(custom_sync_domains=None)
        self.select_tag = select_tag

    def select(
        self,
        _: List[AbstractProcessModel],
        proc_models: List[PyLoihiProcessModel]
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

    def __getitem__(self, id_: int) -> Tuple[np.ndarray, int]:
        data = np.arange(np.prod(self.shape)).reshape(self.shape) + id_
        data = data % np.prod(self.shape)
        label = id_
        return data, label


class SpikeDataset(DummyDataset):
    def __getitem__(self, id_: int) -> Tuple[np.ndarray, int]:
        data = np.arange(np.prod(self.shape)).reshape(self.shape[::-1]) + id_
        data = data.transpose(np.arange(len(self.shape))[::-1]) % 13
        data = data >= 10
        label = id_
        return data, label


class DummyProc(AbstractProcess):
    def __init__(self, shape: tuple) -> None:
        super().__init__()
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


class TestStateDataloader(unittest.TestCase):
    def test_state_loader(self) -> None:
        """Tests state dataloader"""
        num_steps = 30
        shape = (5, 7)
        interval = 5
        offset = 2

        proc = DummyProc(shape)
        dataloader = StateDataloader(dataset=DummyDataset(shape),
                                     interval=interval,
                                     offset=offset)

        ground_truth = ReceiveProcess(
            shape=dataloader.ground_truth.shape, buffer=num_steps
        )

        out = ReceiveProcess(shape=shape, buffer=num_steps)
        dataloader.connect_var(proc.state)
        dataloader.ground_truth.connect(ground_truth.a_in)
        proc.s_out.connect(out.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        proc.run(condition=run_condition, run_cfg=run_config)
        ground_truth_data = ground_truth.data.get()
        out_data = out.data.get()
        proc.stop()

        dataset = DummyDataset(shape)
        for i in range(offset + 1, num_steps):
            id = (i - offset - 1) // interval
            data, ground_truth = dataset[id]
            self.assertTrue(
                np.array_equal(ground_truth, ground_truth_data[..., i].item())
            )
            self.assertTrue(
                np.array_equal(data, out_data[..., i]),
                f'Expected data and out_data at {i=} to be same. '
                f'Found {data=} and {out_data[..., i]=}'
            )


class TestSpikeDataloader(unittest.TestCase):
    def run_test(
        self,
        shape: tuple,
        steps: int,
        interval: int,
        offset: int,
        num_steps: int,
    ) -> None:
        dataloader = SpikeDataloader(
            dataset=SpikeDataset(shape + (steps,)),
            interval=interval,
            offset=offset
        )

        ground_truth = ReceiveProcess(
            shape=dataloader.ground_truth.shape, buffer=num_steps
        )

        out = ReceiveProcess(shape=shape, buffer=num_steps)
        dataloader.ground_truth.connect(ground_truth.a_in)
        dataloader.s_out.connect(out.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        dataloader.run(condition=run_condition, run_cfg=run_config)
        ground_truth_data = ground_truth.data.get()
        out_data = out.data.get()
        dataloader.stop()

        dataset = SpikeDataset(shape + (steps,))
        for i in range(offset + 1, num_steps, interval):
            id = (i - offset - 1) // interval
            data, ground_truth = dataset[id]

            ground_truth_error = np.abs(
                ground_truth_data[..., i:i + interval] - ground_truth
            ).sum()

            if steps > interval:
                spike_error = np.abs(
                    out_data[..., i:i + interval] - data[..., :interval]
                ).sum()
            else:
                spike_error = np.abs(out_data[..., i:i + steps] - data).sum()

            self.assertTrue(ground_truth_error == 0)
            self.assertTrue(
                spike_error == 0,
                f'Expected data and out_data at {i=} to be same. '
                f'Found {data=} and {out_data[..., i:i + steps]=}'
            )

            if steps < interval:
                gap_error = np.abs(out_data[..., i + steps:i + interval]).sum()
                self.assertTrue(gap_error == 0)

    def test_spike_loader(self) -> None:
        """Tests spike dataloader"""
        shape = (5, 7)
        steps = 5
        interval = 5
        offset = 2
        num_steps = 31 + offset
        self.run_test(shape, steps, interval, offset, num_steps)

    def test_spike_loader_less_steps(self) -> None:
        """Tests spike dataloader when data load interval is less than sample
        time steps"""
        shape = (5, 7)
        steps = 3
        interval = 5
        offset = 2
        num_steps = 31 + offset
        self.run_test(shape, steps, interval, offset, num_steps)

    def test_spike_loader_more_steps(self) -> None:
        """Tests spike dataloader when data load interval is less than sample
        time steps."""
        shape = (5, 7)
        steps = 8
        interval = 5
        offset = 2
        num_steps = 31 + offset

        self.run_test(shape, steps, interval, offset, num_steps)


if __name__ == '__main__':
    pass
