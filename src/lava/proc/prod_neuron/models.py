# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2023 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.proc.prod_neuron.process import ProdNeuron


class AbstractProdNeuronModel(PyLoihiProcessModel):
    """Abstract implementation of ProdNeuron.

    Specific implementations inherit from here.
    """
    a_in1 = None
    a_in2 = None
    s_out = None

    vth = None
    exp = None
    v = None

    def run_spk(self) -> None:
        a_in_data1 = self.a_in1.recv()
        a_in_data2 = self.a_in2.recv()

        v = a_in_data1 * a_in_data2
        v /= 2**self.exp

        is_spike = np.abs(v) > self.vth
        sp_out = v * is_spike

        self.s_out.send(sp_out)


@implements(proc=ProdNeuron, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProdNeuornModelFloat(AbstractProdNeuronModel):
    """Floating point implementation of ProdNeuron"""
    a_in1 = LavaPyType(PyInPort.VEC_DENSE, float)
    a_in2 = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: np.ndarray = LavaPyType(np.ndarray, float)
    exp: np.ndarray = LavaPyType(np.ndarray, float)


@implements(proc=ProdNeuron, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyProdNeuronModelFixed(AbstractProdNeuronModel):
    """Fixed point implementation of ProdNeuron"""
    a_in1 = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    a_in2 = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
