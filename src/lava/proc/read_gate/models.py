# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.proc.read_gate.process import ReadGate


@implements(ReadGate, protocol=LoihiProtocol)
@requires(CPU)
class ReadGatePyModel(PyLoihiProcessModel):
    """CPU model for the ReadGate process.

    The model verifies if better payload (cost) has been notified by the
    downstream processes, if so, it reads those processes state and sends out to
    the upstream process the new payload (cost) and the network state.
    """
    target_cost: int = LavaPyType(int, np.int32, 32)
    cost_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                   precision=32)
    acknowledgemet: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                          precision=32)
    cost_out: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.int32, precision=32
    )
    solution_out: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.int32, precision=32
    )
    send_pause_request: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.int32, precision=32
    )
    solution_reader = LavaPyType(PyRefPort.VEC_DENSE, np.int32,
                                 precision=32)
    min_cost: int = None
    solution: np.ndarray = None

    def post_guard(self):
        """Decide whether to run post management phase."""
        if self.min_cost:
            return True
        return False

    def run_spk(self):
        """Execute spiking phase, integrate input, update dynamics and
        send messages out."""
        cost = self.cost_in.recv()
        if cost[0]:
            self.min_cost = cost[0]
            print("Found a solution with cost: ", self.min_cost)
            self.cost_out.send(np.asarray([0]))
            self.send_pause_request.send(np.asarray([0]))
        elif self.solution is not None:
            self.solution_out.send(self.solution)
            self.cost_out.send(np.asarray([self.min_cost]))
            self.solution = None
            self.min_cost = None
            self.send_pause_request.send(np.asarray([0]))
        else:
            self.cost_out.send(np.asarray([0]))
            self.send_pause_request.send(np.asarray([0]))

    def run_post_mgmt(self):
        """Execute post management phase."""
        self._req_pause = True
        self.solution = self.solution_reader.read()
        self._req_pause = False
