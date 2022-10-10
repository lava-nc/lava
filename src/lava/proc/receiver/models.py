# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.proc.receiver.process import Receiver


@implements(proc=Receiver, protocol=LoihiProtocol)
@requires(CPU)
class ReceiverModel(PyLoihiProcessModel):
    """CPU model for the Receiver process.

    The process saves any accumulated input messages as a payload variable.
    """
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    payload: np.ndarray = LavaPyType(np.ndarray, int, 32)

    def run_spk(self):
        """Execute spiking phase, integrate incomming input and update
        payload."""
        synaptic_input = self.a_in.recv()
        self.payload[:] = synaptic_input
