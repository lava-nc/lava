# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class RateReader(AbstractProcess):
    """
    Process that stores recent spikes in a buffer and computes the spike rate
    at each timestep.
    """
    def __init__(self, shape, buffer_size, num_steps=1):
        super().__init__(shape=shape,
                         buffer_size=buffer_size,
                         num_steps=num_steps)
        self.in_port = InPort(shape)
        self.buffer = Var(shape=shape + (buffer_size,))
        self.rate = Var(shape=shape, init=0)
        self.out_port = OutPort(shape)


@implements(proc=RateReader, protocol=LoihiProtocol)
@requires(CPU)
class PyRateReaderPMDense(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, 8)
    buffer: np.ndarray = LavaPyType(np.ndarray, np.int32)
    rate: np.ndarray = LavaPyType(np.ndarray, float)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._buffer_size = proc_params["buffer_size"]

    def post_guard(self):
        # Ensures that run_post_mgmt runs after run_spk at every
        # time step.
        return True

    def run_post_mgmt(self):
        # Runs after run_spk in every time step and computes the
        # spike rate from the buffer.
        spikes = self.in_port.recv()
        self.buffer[..., (self.time_step - 1) % self._buffer_size] = spikes
        self.rate = np.mean(self.buffer, axis=-1)

    def run_spk(self):
        self.out_port.send(self.rate)
