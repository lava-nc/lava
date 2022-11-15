# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class RF(AbstractProcess):
    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 period: float,
                 alpha: float,
                 state_exp: ty.Optional[int] = 0,
                 decay_bits: ty.Optional[int] = 0,
                 vth: ty.Optional[float] = 1,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None):
        """Resonate and Fire (RF) neural Process.

        RF dynamics abstracts to:
        Re[t] = (1 - a) * (cos(theta)* Re[t-1] - sin(theta) * Im[t-1]) + re_inp
        Im[t] = (1 - a) * (sin(theta)* Re[t-1] + cos(theta) * Im[t-1]) + im_inp
        s[t] = (Re[t] >= vth) & (Im[t] >= 0) & (Im[t -1] < 0)

        Re[t]: real component/voltage
        Im[t]: imaginary component/voltage
        re_inp: real input at timestep t
        im_inp: imag input at timestep t
        a: alpha decay
        s[t]: output spikes

        Parameters
        ----------
        shape : tuple(int)
            Number and topology of RF neurons.
        period : float, list, numpy.ndarray, optional
            Neuron's internal resonator frequency
        alpha : float, list, numpy.ndarray, optional
            Decay real and imaginary voltage
        state_exp : int, list, numpy.ndarray, optional
            Scaling exponent with base 2 for the spike message.
            Note: This should only be used for fixed point models.
            Default is 0.
        decay_bits : float, list, numpy.ndarray, optional
            Desired bit precision of neuronal decay
            Default is 0.
        vth : float, optional
            Neuron threshold voltage, exceeding which, the neuron will spike.
            Currently, only a single threshold can be set for the entire
            population of neurons.

        Example
        -------
        >>> rf = RF(shape=(200, 15), period=10, alpha=.07)
        This will create 200x15 RF neurons that all have the same period decay
        of 10 and alpha decay of .07
        """
        sin_decay = (1 - alpha) * np.sin(np.pi * 2 * 1 / period)
        cos_decay = (1 - alpha) * np.cos(np.pi * 2 * 1 / period)
        super().__init__(shape=shape, sin_decay=sin_decay, cos_decay=cos_decay,
                         state_exp=state_exp, decay_bits=decay_bits, vth=vth,
                         name=name, log_config=log_config)

        if state_exp > 0:
            vth = int(vth * (1 << state_exp))
        if decay_bits > 0:
            sin_decay = np.int32(sin_decay * (1 << decay_bits))
            cos_decay = np.int32(cos_decay * (1 << decay_bits))

        self.a_real_in = InPort(shape=shape)
        self.a_imag_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.real = Var(shape=shape, init=0)
        self.imag = Var(shape=shape, init=0)
        self.sin_decay = Var(shape=(1,), init=sin_decay)
        self.cos_decay = Var(shape=(1,), init=cos_decay)
        self.state_exp = Var(shape=(1,), init=state_exp)
        self.decay_bits = Var(shape=(1,), init=decay_bits)
        self.vth = Var(shape=(1,), init=vth)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']
