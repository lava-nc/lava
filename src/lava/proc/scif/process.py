# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class SCIF(AbstractProcess):

    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            bias: ty.Optional[int] = 1,
            theta: ty.Optional[int] = 4,
            neg_tau_ref: ty.Optional[int] = -5) -> None:
        """
        Stochastic Constraint Integrate and Fire neuron Process.

        Parameters
        ----------
        shape: Tuple
            shape of the sigma process. Default is (1,).
        bias: int
            bias current driving the SCIF neuron. Default is 1 (arbitrary).
        theta: int
            threshold above which a SCIF neuron would fire winner-take-all
            spike. Default is 4 (arbitrary).
        neg_tau_ref: int
            refractory time period (in number of algorithmic time-steps) for a
            SCIF neuron after firing winner-take-all spike. Default is -5 (
            arbitrary).
        """
        super().__init__(shape=shape)

        self.a_in = InPort(shape=shape)
        self.s_sig_out = OutPort(shape=shape)
        self.s_wta_out = OutPort(shape=shape)

        self.u = Var(shape=shape, init=np.zeros(shape=shape).astype(int))
        self.v = Var(shape=shape, init=np.zeros(shape=shape).astype(int))
        self.beta = Var(shape=shape, init=np.zeros(shape=shape).astype(int))
        self.enable_noise = Var(shape=shape, init=np.zeros(
            shape=shape).astype(int))

        self.bias = Var(shape=shape, init=int(bias))
        self.theta = Var(shape=(1,), init=int(theta))
        self.neg_tau_ref = Var(shape=(1,), init=int(neg_tau_ref))

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self.proc_params['shape']
