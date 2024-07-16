# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class ProdNeuron(AbstractProcess):
    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            vth: ty.Optional[int] = 1,
            exp: ty.Optional[int] = 0) -> None:
        """ProdNeuron

        Multiplies two graded inputs and outputs result as graded spike.

        v[t] = (a_in1 * a_in2) >> exp
        s_out = v[t] * (|v[t]| > vth)

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
