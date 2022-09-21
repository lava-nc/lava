# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort


# Ring Buffer
class RingBuffer(AbstractProcess):
    """Spike generator process from circular data buffer.

    Parameters
    ----------
    data: np array
        data to generate spike from. Last dimension is assumed as time.
    """
    def __init__(self,
                 *,
                 data: np.ndarray) -> None:
        super().__init__(data=data)
        self.data = Var(shape=data.shape, init=data)
        self.s_out = OutPort(shape=data.shape[:-1])  # last dimension is time


class AbstractPyRingBuffer(PyLoihiProcessModel):
    """Abstract ring buffer process model."""
    s_out = None
    data = None

    def run_spk(self) -> None:
        buffer = self.data.shape[-1]
        self.s_out.send(self.data[..., (self.time_step - 1) % buffer])


@implements(proc=RingBuffer, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySendModelFloat(AbstractPyRingBuffer):
    """Float ring buffer send process model."""
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)


@implements(proc=RingBuffer, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PySendModelFixed(AbstractPyRingBuffer):
    """Fixed point ring buffer send process model."""
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32)
