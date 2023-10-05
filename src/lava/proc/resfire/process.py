# Copyright (C) 2022-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class RFZero(AbstractProcess):
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 freqs: np.ndarray,  # Hz
                 decay_tau: np.ndarray,  # seconds
                 dt: ty.Optional[float] = 0.001,  # seconds/timestep
                 uth: ty.Optional[int] = 1) -> None:
        """
        RFZero
        Resonate and fire neuron with spike trigger of threshold and
        0-phase crossing. Graded spikes carry amplitude of oscillation.

        Parameters
        ----------
        shape : tuple(int)
            Number and topology of RF neurons.
        freqs : numpy.ndarray
            Frequency for each neuron (Hz).
        decay_tau : numpy.ndarray
            Decay time constant (s).
        dt : float, optional
            Time per timestep. Default is 0.001 seconds.
        uth : float, optional
            Neuron threshold voltage.
            Currently, only a single threshold can be set for the entire
            population of neurons.
        """
        super().__init__(shape=shape)

        self.u_in = InPort(shape=shape)
        self.v_in = InPort(shape=shape)

        self.s_out = OutPort(shape=shape)

        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)

        ll = -1 / decay_tau

        lct = np.array(np.exp(dt * ll) * np.cos(dt * freqs * np.pi * 2))
        lst = np.array(np.exp(dt * ll) * np.sin(dt * freqs * np.pi * 2))
        lct = (lct * 2**15).astype(np.int32)
        lst = (lst * 2**15).astype(np.int32)

        self.lct = Var(shape=shape, init=lct)
        self.lst = Var(shape=shape, init=lst)
        self.uth = Var(shape=(1,), init=uth)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']
