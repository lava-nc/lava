# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from numpy import typing as npty

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class AbstractScif(AbstractProcess):
    """Abstract Process for Stochastic Constraint Integrate-and-Fire
    (SCIF) neurons.
    """

    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            step_size: ty.Optional[int] = 1,
            theta: ty.Optional[int] = 4,
            neg_tau_ref: ty.Optional[int] = -5) -> None:
        """
        Stochastic Constraint Integrate and Fire neuron Process.

        Parameters
        ----------
        shape: Tuple
            Number of neurons. Default is (1,).
        step_size: int
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

        self.cnstr_intg = Var(shape=shape, init=np.zeros(shape=shape).astype(
            int))
        self.state = Var(shape=shape, init=np.zeros(shape=shape).astype(int))
        self.spk_hist = Var(shape=shape, init=np.zeros(shape=shape).astype(int))
        self.noise_ampl = Var(shape=shape, init=np.zeros(
            shape=shape).astype(int))

        self.step_size = Var(shape=shape, init=int(step_size))
        self.theta = Var(shape=(1,), init=int(theta))
        self.neg_tau_ref = Var(shape=(1,), init=int(neg_tau_ref))

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self.proc_params['shape']


class CspScif(AbstractScif):
    """Stochastic Constraint Integrate-and-Fire neurons to solve CSPs.
    """

    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...],
                 step_size: ty.Optional[int] = 1,
                 theta: ty.Optional[int] = 4,
                 neg_tau_ref: ty.Optional[int] = -5):

        super(CspScif, self).__init__(shape=shape,
                                      step_size=step_size,
                                      theta=theta,
                                      neg_tau_ref=neg_tau_ref)


class QuboScif(AbstractScif):
    """Stochastic Constraint Integrate-and-Fire neurons to solve QUBO
    problems.
    """

    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...],
                 cost_diag: npty.NDArray,
                 step_size: ty.Optional[int] = 1,
                 theta: ty.Optional[int] = 4,
                 neg_tau_ref: ty.Optional[int] = -5):

        super(QuboScif, self).__init__(shape=shape,
                                       step_size=step_size,
                                       theta=theta,
                                       neg_tau_ref=neg_tau_ref)
        self.cost_diagonal = Var(shape=shape, init=cost_diag)
