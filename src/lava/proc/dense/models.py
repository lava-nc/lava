# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.dense.process import Dense


@implements(proc=Dense, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyDenseModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    # previously hidden var
    weights: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=8)

    def run_spk(self):
        s_in = self.s_in.recv()
        a_out = self.weights[:, s_in].sum(axis=1)
        self.a_out.send(a_out)
        self.a_out.flush()

    def run_lrn(self):
        pass
