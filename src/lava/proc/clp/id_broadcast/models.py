# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.proc.clp.id_broadcast.process import IdBroadcast


@implements(proc=IdBroadcast, protocol=LoihiProtocol)
@requires(CPU)
class IdBroadcastModel(PyLoihiProcessModel):
    """CPU model for the IdBroadcast process.

    The process sends out a graded spike with payload equal to a_in.
    """
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    val: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def run_spk(self):
        """Execute spiking phase, send value of a_in."""

        a_in_data = self.a_in.recv()
        self.val = a_in_data
        self.s_out.send(a_in_data)
