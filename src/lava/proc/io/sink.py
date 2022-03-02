# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from typing import Any, Dict, Tuple, Union

from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, RefPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyRefPort


# Ring Buffer
class RingBuffer(AbstractProcess):
    """Process for receiving arbitrarily shaped data into a ring buffer
    memory. Works as a substitute for probing.

    Parameters
    ----------
    shape: tuple
        shape of the process
    buffer: int
        size of data sink buffer
    """
    def __init__(self, **kwargs: Union[int, Tuple[int, ...]]) -> None:
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (1,))
        buffer = kwargs.get('buffer')
        self.shape = shape
        self.a_in = InPort(shape=shape)
        buffer_shape = shape + (buffer,)
        self.data = Var(shape=buffer_shape, init=np.zeros(buffer_shape))


class AbstractPyReceiveModel(PyLoihiProcessModel):
    """Abstract ring buffer receive process model."""
    a_in = None
    data = None

    def run_spk(self) -> None:
        """Receive spikes and store in an internal variable"""
        data = self.a_in.recv()
        buffer = self.data.shape[-1]
        self.data[..., (self.time_step - 1) % buffer] = data


@implements(proc=RingBuffer, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyReceiveModelFloat(AbstractPyReceiveModel):
    """Float ring buffer receive process model."""
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)


@implements(proc=RingBuffer, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyReceiveModelFixed(AbstractPyReceiveModel):
    """Fixed point ring buffer receive process model."""
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)


# Read
class Read(AbstractProcess):
    """Reads and logs the data of it's internal state at a
    set interval and offset (phase).

    Parameters
    ----------
    buffer: int
        number of samples to buffer
    interval : int, optional
        reset interval, by default 1
    offset : int, optional
        reset offset (phase), by default 0
    """
    def __init__(
        self,
        buffer: int,
        interval: int = 1,
        offset: int = 0,
    ) -> None:
        super().__init__()
        self.interval = Var((1,), interval)
        self.offset = Var((1,), offset % interval)
        self.buffer = buffer
        self.state = RefPort((1,))
        buffer_shape = (1,) + (self.buffer,)
        self.data = Var(shape=buffer_shape, init=np.zeros(buffer_shape))

    def connect_var(self, var: Var) -> None:
        self.state = RefPort(var.shape)
        self.state.connect_var(var)
        buffer_shape = var.shape + (self.buffer,)
        self.data = Var(shape=buffer_shape, init=np.zeros(buffer_shape))
        self._post_init()


class AbstractPyRead(PyLoihiProcessModel):
    """Abstract Read Var process implementation."""
    state: Union[PyRefPort, None] = None
    data = None
    interval: np.ndarray = LavaPyType(np.ndarray, int)
    offset: np.ndarray = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params: Dict[str, Any]) -> None:
        super().__init__(proc_params)
        self.counter = 0

    def post_guard(self) -> None:
        return (self.time_step - 1) % self.interval == self.offset

    def run_post_mgmt(self) -> None:
        data = self.state.read()
        buffer = self.data.shape[-1]
        self.data[..., self.counter % buffer] = data
        self.counter += 1


@implements(proc=Read, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyReadFixed(AbstractPyRead):
    """Read Var process implementation for int type."""
    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)


@implements(proc=Read, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyReadFloat(AbstractPyRead):
    """Read Var process implementation for float type."""
    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)
