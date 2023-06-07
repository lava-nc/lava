# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.proc.spiker.process import Spiker


@implements(proc=Spiker, protocol=LoihiProtocol)
@requires(CPU)
class SpikerModel(PyLoihiProcessModel):
    """CPU model for the Spiker process.

    The process sends messages at the specified rate with a specified payload.
    """
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    rate: np.ndarray = LavaPyType(np.ndarray, int, 32)
    counter: np.ndarray = LavaPyType(np.ndarray, int, 32)
    payload: np.ndarray = LavaPyType(np.ndarray, int, 32)

    def run_spk(self):
        """Execute spiking phase, send a payload at the pre-determined rate."""

        condition = (self.counter == self.rate)

        self.s_out.send(np.where(condition,
                                 self.payload,
                                 0)
                        )
        self.counter = np.where(condition,
                                1,
                                self.counter + 1)
