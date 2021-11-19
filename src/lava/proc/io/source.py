# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel


class SendProcess(AbstractProcess):
    """Spike generator process

    Parameters
    ----------
    data: np array
        data to generate spike from. Last dimension is assumed as time.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        data = kwargs.pop('data')
        self.data = Var(shape=data.shape, init=data)
        self.s_out = OutPort(shape=data.shape[:-1])  # last dimension is time


class AbstractPySendModel(PyLoihiProcessModel):
    """Template send process model."""
    s_out = None
    data = None

    def run_spk(self):
        buffer = self.data.shape[-1]
        self.s_out.send(self.data[..., (self.current_ts - 1) % buffer])


@implements(proc=SendProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySendModelFloat(AbstractPySendModel):
    """Float send process model."""
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)


@implements(proc=SendProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PySendModelFixed(AbstractPySendModel):
    """Fixed point send process model."""
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    data: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
