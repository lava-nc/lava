# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import Iterable, Union
import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort, RefPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import HostCPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel


# State dataloader ###########################################################
class StateDataloader(AbstractProcess):
    """Dataloader object that loads new data sample to internal state at a
    set interval and offset (phase).

    Parameters
    ----------
    dataset : Iterable
        The actual dataset object. Dataset is expected to return
        ``(input, label/ground_truth)`` when indexed.
    interval : int, optional
        Interval between each data load, by default 1
    offset : int, optional
        Offset (phase) for each data load, by default 0
    """
    def __init__(
        self,
        dataset: Iterable,
        interval: int = 1,
        offset: int = 0,
    ) -> None:
        super().__init__()
        self.interval = Var((1,), interval)
        self.offset = Var((1,), offset % interval)
        data, gt = dataset[0]
        self.state = RefPort(data.shape)
        self.gt = OutPort((1,) if np.isscalar(gt) else gt.shape)
        self.proc_params['saved_dataset'] = dataset

    def connect_var(self, var: Var) -> None:
        """Connects the internal state (ref-port) to the variable. The variable
        can be internal state of some other process which needs to reflect the
        dataset input.

        Parameters
        ----------
        var : Var
            Variable that is connected to this object's state (ref-port).
        """
        self.state = RefPort(var.shape)
        self.state.connect_var(var)
        self._post_init()


@requires(HostCPU)
class AbstractPyStateModel(PyLoihiProcessModel):
    state: Union[PyRefPort, None] = None
    gt: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    interval: np.ndarray = LavaPyType(np.ndarray, int)
    offset: np.ndarray = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.sample_id = 0
        self.gt_state = 0
        self.dataset = self.proc_params['saved_dataset']

    def gt_array(self) -> np.ndarray:
        if np.isscalar(self.gt_state):
            return np.array([self.gt_state])
        return self.gt_state

    def run_spk(self) -> None:
        self.gt.send(self.gt_array())

    def post_guard(self) -> None:
        return (self.current_ts - 1) % self.interval == self.offset

    def run_post_mgmt(self) -> None:
        data, self.gt_state = self.dataset[self.sample_id]
        self.gt_state = self.gt_array()
        self.state.write(data)
        self.sample_id += 1
        if self.sample_id == len(self.dataset):
            self.sample_id = 0


@implements(proc=StateDataloader, protocol=LoihiProtocol)
@tag('fixed_pt')
class PyStateModelFixed(AbstractPyStateModel):
    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int32)


@implements(proc=StateDataloader, protocol=LoihiProtocol)
@tag('floating_pt')
class PyStateModelFloat(AbstractPyStateModel):
    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)


# Spike Dataloader ############################################################
class SpikeDataloader(AbstractProcess):
    """Dataloader object that sends spike for a input sample at a
    set interval and offset (phase).

    Parameters
    ----------
    dataset : Iterable
        The actual dataset object. Dataset is expected to return
        ``(spike, label/ground_truth)`` when indexed.
    interval : int, optional
        Interval between each data load, by default 1
    offset : int, optional
        Offset (phase) for each data load, by default 0
    """
    def __init__(
        self,
        dataset: Iterable,
        interval: int = 1,
        offset: int = 0,
    ) -> None:
        super().__init__()
        self.interval = Var((1,), interval)
        self.offset = Var((1,), offset % interval)
        data, gt = dataset[0]
        data_shape = data.shape[:-1] + (interval,)
        self.data = Var(shape=data_shape, init=np.zeros(data_shape))
        self.s_out = OutPort(shape=data.shape[:-1])  # last dimension is time
        self.gt = OutPort(shape=(1,) if np.isscalar(gt) else gt.shape)
        self.proc_params['saved_dataset'] = dataset


@requires(HostCPU)
class AbstractPySpikeModel(PyLoihiProcessModel):
    s_out: Union[PyOutPort, None] = None
    gt: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    data: Union[np.ndarray, None] = None
    interval: np.ndarray = LavaPyType(np.ndarray, int)
    offset: np.ndarray = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.sample_id = 0
        self.sample_time = 0
        self.gt_state = 0
        self.dataset = self.proc_params['saved_dataset']

    def gt_array(self) -> np.ndarray:
        if np.isscalar(self.gt_state):
            return np.array([self.gt_state])
        return self.gt_state

    def run_spk(self) -> None:
        self.s_out.send(self.data[..., self.sample_time % self.interval.item()])
        self.gt.send(self.gt_array())
        self.sample_time += 1

    def post_guard(self) -> None:
        return (self.current_ts - 1) % self.interval == self.offset

    def run_post_mgmt(self) -> None:
        data, self.gt_state = self.dataset[self.sample_id]
        self.gt_state = self.gt_array()
        if data.shape[-1] >= self.interval:
            self.data = data[..., :self.interval.item()]
        else:
            self.data = np.zeros_like(self.data)
            self.data[..., :data.shape[-1]] = data
        self.sample_time = 0
        self.sample_id += 1
        if self.sample_id == len(self.dataset):
            self.sample_id = 0


@implements(proc=SpikeDataloader, protocol=LoihiProtocol)
@tag('fixed_pt')
class PySpikeModelFixed(AbstractPySpikeModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)


@implements(proc=SpikeDataloader, protocol=LoihiProtocol)
@tag('floating_pt')
class PySpikeModelFloat(AbstractPySpikeModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)
