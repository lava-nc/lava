# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import numpy as np
from lava.proc.resfire.process import RFZero

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel


@implements(proc=RFZero, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyRFZeroModelFixed(PyLoihiProcessModel):
    """Fixed point implementation of RFZero"""
    u_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    v_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)

    uth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    u: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    lst: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    lct: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def run_spk(self) -> None:
        u_in = self.u_in.recv()
        v_in = self.v_in.recv()

        new_u = ((self.u * self.lct) // (2**15)
                 - (self.v * self.lst) // (2**15) + u_in)

        new_v = ((self.v * self.lct) // (2**15)
                 + (self.u * self.lst) // (2**15) + v_in)

        s_out = new_u * (new_u > self.uth) * (new_v >= 0) * (self.v < 0)

        self.u = new_u
        self.v = new_v

        self.s_out.send(s_out)
