# Copyright (C) 2024 Intel Corporation
# Copyright (C) 2024 Jannik Luboeinski
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class ATRLIF(AbstractProcess):
    """
    Adaptive Threshold and Refractoriness Leaky-Integrate-and-Fire Process.
    With activation input port `a_in` and spike output port `s_out`.

    Note that non-integer parameter values are supported, but can lead to
    deviating results in models that employ fixed-point computation.

    Dynamics (cf. https://github.com/lava-nc/lava-dl/blob/main/src/lava/lib/dl/
                          slayer/neuron/alif.py,
                  https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/
                          lib/dl/slayer/neuron_dynamics/dynamics.ipynb):
      i[t] = (1-delta_i)*i[t-1] + x[t]
      v[t] = (1-delta_v)*v[t-1] + i[t] + bias
      theta[t] = (1-delta_theta)*(theta[t-1] - theta_0) + theta_0
      r[t] = (1-delta_r)*r[t-1]

    Spike event:
      s[t] = (v[t] - r[t]) >= theta[t]

    Post spike event:
      r[t] = r[t] + 2*theta[t]
      theta[t] = theta[t] + theta_step

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    i : float, list, numpy.ndarray, optional
        Initial value of the neuron's current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neuron's voltage (membrane potential).
    theta : float, list, numpy.ndarray, optional
        Initial value of the threshold
    r : float, list, numpy.ndarray, optional
        Initial value of the refractory state
    s : bool, list, numpy.ndarray, optional
        Initial spike state
    delta_i : float, optional
        Decay constant for current i.
    delta_v : float, optional
        Decay constant for voltage v.
    delta_theta : float, optional
        Decay constant for threshold theta.
    delta_r : float, optional
        Decay constant for refractory state r.
    theta_0 : float, optional
        Initial/baselien value of threshold theta.
    theta_step : float, optional
        Increase of threshold theta upon the occurrence of a spike.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of the neuron's bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of the neuron's bias, if needed. Mostly for fixed-point
        implementations. Ignored for floating-point implementations.

    Example
    -------
    >>> atrlif = ATRLIF(shape=(200, 15), decay_theta=10, decay_v=5)
    This will create 200x15 ATRLIF neurons that all have the same threshold
    decay of 10 and voltage decay of 5.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        i: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        theta: ty.Optional[ty.Union[float, list, np.ndarray]] = 5,
        r: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        s: ty.Optional[ty.Union[bool, list, np.ndarray]] = 0,
        delta_i: ty.Optional[float] = 0.4,
        delta_v: ty.Optional[float] = 0.4,
        delta_theta: ty.Optional[float] = 0.2,
        delta_r: ty.Optional[float] = 0.2,
        theta_0: ty.Optional[float] = 5,
        theta_step: ty.Optional[float] = 3.75,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None
    ) -> None:

        super().__init__(
            shape=shape,
            i=i,
            v=v,
            theta=theta,
            r=r,
            s=s,
            delta_i=delta_i,
            delta_v=delta_v,
            delta_theta=delta_theta,
            delta_r=delta_r,
            theta_0=theta_0,
            theta_step=theta_step,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config
        )

        # Ports
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        # Bias
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.bias_exp = Var(shape=shape, init=bias_exp)

        # Variables
        self.i = Var(shape=shape, init=i)
        self.v = Var(shape=shape, init=v)
        self.theta = Var(shape=shape, init=theta)
        self.r = Var(shape=shape, init=r)
        self.s = Var(shape=shape, init=s)

        # Parameters
        self.delta_i = Var(shape=(1,), init=delta_i)
        self.delta_v = Var(shape=(1,), init=delta_v)
        self.delta_theta = Var(shape=(1,), init=delta_theta)
        self.delta_r = Var(shape=(1,), init=delta_r)
        self.theta_0 = Var(shape=(1,), init=theta_0)
        self.theta_step = Var(shape=(1,), init=theta_step)
