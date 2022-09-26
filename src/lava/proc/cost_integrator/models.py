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

from lava.proc.cost_integrator.process import CostIntegrator


@implements(proc=CostIntegrator, protocol=LoihiProtocol)
@requires(CPU)
class CostIntegratorModel(PyLoihiProcessModel):
    """CPU model for the CostIntegrator process.

    The process adds up local cost components from downstream units comming as
    spike payload. It has a min_cost variable which keeps track of the best
    cost seen so far, if the new cost is better, the minimum cost is updated
    and send as an output spike to an upstream process.
    """
    cost_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    update_buffer: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    min_cost: np.ndarray = LavaPyType(np.ndarray, int, 32)
    cost: np.ndarray = LavaPyType(np.ndarray, int, 32)

    def run_spk(self):
        """Execute spiking phase, integrate input, update dynamics and send
        messages out."""
        cost = self.cost_in.recv()
        if cost < self.min_cost:
            self.min_cost[:] = cost
            self.update_buffer.send(cost)
        else:
            self.update_buffer.send(np.asarray([0]))
