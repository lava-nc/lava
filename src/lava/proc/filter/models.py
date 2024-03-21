# Copyright (C) 2021-24 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.filter.process import ExpFilter 

@implements(proc=ExpFilter, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyExpFilterModelFloat(PyLoihiProcessModel):
    """Implementation of Exponential Filter process in floating
    point precision. This short and simple ProcessModel can be used for quick
    algorithmic prototyping, without engaging with the nuances of a fixed
    point implementation.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    value: np.ndarray = LavaPyType(np.ndarray, float)
    tau: float = LavaPyType(float, float)

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        a_in_data = self.a_in.recv()

        self.value[:] = self.value * (1 - self.tau) + a_in_data

        self.s_out.send(self.value)