# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from scipy.sparse import csr_matrix

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.sparse.process import Sparse

class AbstractPySparseModelFloat(PyLoihiProcessModel):
    """Implementation of Conn Process with Sparse synaptic connections in
    floating point precision. This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation.
    """

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    a_buff: np.ndarray = LavaPyType(np.ndarray, float)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons)in C-order (row major).
    weights: np.ndarray = LavaPyType(csr_matrix, float)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def run_spk(self):
        # The a_out sent on a each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff)
        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
            self.a_buff = self.weights.dot(s_in)
        else:
            s_in = self.s_in.recv().astype(bool)
            # A1: return as flattend array
            self.a_buff = self.weights[:, s_in].sum(axis=1).A1


@implements(proc=Sparse, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PySparseModelFloat(AbstractPySparseModelFloat):
    pass

