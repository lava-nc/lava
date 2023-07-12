# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2023 Intel Corporation.
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
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class ProdNeuron(AbstractProcess):
    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            vth=1,
            exp=0) -> None:
        """ProdNeuron

        Multiplies two graded inputs.

        Parameters
        ----------

        shape : tuple(int)
            Number and topology of ProdNeuron neurons.
        vth : int
            Threshold
        exp : int
            Fixed-point base
        """
        super().__init__(shape=shape)

        self.a_in1 = InPort(shape=shape)
        self.a_in2 = InPort(shape=shape)

        self.s_out = OutPort(shape=shape)

        self.vth = Var(shape=(1,), init=vth)
        self.exp = Var(shape=(1,), init=exp)

        self.v = Var(shape=shape, init=np.zeros(shape, 'int32'))

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']
