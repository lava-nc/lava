# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel


class ReceiveProcess(AbstractProcess):
    """Receive process

    Parameters
    ----------
    shape: tuple
        shape of the process
    buffer: int
        size of data sink buffer
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (1,))
        buffer = kwargs.get('buffer')
        self.shape = shape
        self.a_in = InPort(shape=shape)
        buffer_shape = shape + (buffer,)
        self.data = Var(shape=buffer_shape, init=np.zeros(buffer_shape))


class AbstractPyReceiveModel(PyLoihiProcessModel):
    """Template receive process model."""
    a_in = None
    data = None

    def run_spk(self):
        """Receive spikes and store in an internal variable"""
        data = self.a_in.recv()
        buffer = self.data.shape[-1]
        self.data[..., (self.current_ts - 1) % buffer] = data


@implements(proc=ReceiveProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyReceiveModelFloat(AbstractPyReceiveModel):
    """Float receive process model."""
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)


@implements(proc=ReceiveProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyReceiveModelFixed(AbstractPyReceiveModel):
    """Fixed point receive process model."""
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)
