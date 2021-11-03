# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements_protocol, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


@implements_protocol(protocol=LoihiProtocol)
@requires(CPU)
class PyLifModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    bias: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=12)
    du: int = LavaPyType(int, np.uint16, precision=12)
    dv: int = LavaPyType(int, np.uint16, precision=12)
    vth: int = LavaPyType(int, int, precision=8)

    def run_spk(self):
        self.u[:] = self.u * ((2 ** 12 - self.du) // 2 ** 12)
        a_in_data = self.a_in.recv()
        self.u[:] += a_in_data
        self.v[:] = self.v * \
            ((2 ** 12 - self.dv) // 2 ** 12) + self.u + self.bias
        s_out = self.v > self.vth
        self.v[s_out] = 0  # Reset voltage to 0. This is Loihi-1 compatible.
        self.s_out.send(s_out)
