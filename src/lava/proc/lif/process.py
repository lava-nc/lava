# Copyright (C) 2022-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
from lava.magma.core.process.process import LogConfig, AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.neuron import LearningNeuronProcess


class AbstractLIF(AbstractProcess):
    """Abstract class for variables common to all neurons with leaky
    integrator dynamics."""

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Union[float, list, np.ndarray],
        v: ty.Union[float, list, np.ndarray],
        du: float,
        dv: float,
        bias_mant: ty.Union[float, list, np.ndarray],
        bias_exp: ty.Union[float, list, np.ndarray],
        name: str,
        log_config: LogConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=u)
        self.v = Var(shape=shape, init=v)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.bias_mant = Var(shape=shape, init=bias_mant)


class LIF(AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process.

    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.

    Example
    -------
    >>> lif = LIF(shape=(200, 15), du=10, dv=5)
    This will create 200x15 LIF neurons that all have the same current decay
    of 10 and voltage decay of 5.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        vth: ty.Optional[float] = 10,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self.vth = Var(shape=(1,), init=vth)


class LearningLIF(LearningNeuronProcess, AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process with learning enabled.

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.
    log_config: LogConfig, optional
        Configure the amount of debugging output.
    learning_rule: LearningRule
        Defines the learning parameters and equation.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        vth: ty.Optional[float] = 10,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        learning_rule: Loihi2FLearningRule = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
            learning_rule=learning_rule,
            **kwargs,
        )
        self.vth = Var(shape=(1,), init=vth)


class TernaryLIF(AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process with *ternary* spiking
    output, i.e., +1, 0, and -1 spikes. When the voltage of a TernaryLIF neuron
    exceeds its upper threshold (vth_hi), it issues a positive spike and when
    the voltage drops below its lower threshold (vth_lo), it issues a negative
    spike. Between the two thresholds, the neuron follows leaky linear
    dynamics.

    This class inherits the state variables and ports from AbstractLIF and
    adds two new threshold variables for upper and lower thresholds.

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth_hi : float, optional
        Upper threshold voltage, exceeding which the neuron spikes +1.
        Currently, only a single higher threshold can be set for the entire
        population of neurons.
    vth_lo : float, optional
        Lower threshold voltage, below which the neuron spikes -1.
        Currently, only a single lower threshold can be set for the entire
        population of neurons.

    See Also
    --------
    lava.proc.lif.process.LIF: 'Regular' leaky-integrate-and-fire neuron for
    documentation on rest of the parameters.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        vth_hi: ty.Optional[float] = 10,
        vth_lo: ty.Optional[float] = -10,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
        )

        if np.isscalar(vth_lo) and np.isscalar(vth_hi) and vth_lo > vth_hi:
            raise ValueError(
                f"The lower threshold (vth_lo) must be"
                f"smaller than the higher threshold (vth_hi)."
                f"Got vth_lo={vth_lo}, vth_hi={vth_hi}."
            )
        self.vth_hi = Var(shape=(1,), init=vth_hi)
        self.vth_lo = Var(shape=(1,), init=vth_lo)


class LIFReset(LIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process that resets its internal
    states in regular intervals.

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.
    reset_interval : int, optional
        The interval of neuron state reset. By default 1 timestep.
    reset_offset : int, optional
        The phase/offset of neuron reset. By defalt at 0th timestep.


    See Also
    --------
    lava.proc.lif.process.LIF: 'Regular' leaky-integrate-and-fire neuron for
    documentation on rest of the behavior between reset intervals.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        vth: ty.Optional[float] = 10,
        reset_interval: ty.Optional[int] = 1,
        reset_offset: ty.Optional[int] = 0,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            vth=vth,
            name=name,
            log_config=log_config,
        )
        if reset_interval < 1:
            raise ValueError("Reset interval must be > 0.")
        if reset_offset < 0:
            raise ValueError("Reset offset must be positive.")

        self.proc_params["reset_interval"] = reset_interval
        self.proc_params["reset_offset"] = reset_offset


class LIFRefractory(LIF):

    """Leaky-Integrate-and-Fire (LIF) process with refractory period.

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.
    refractory_period : int, optional
        The interval of the refractory period. 1 timestep by default.


    See Also
    --------
    lava.proc.lif.process.LIF: 'Regular' leaky-integrate-and-fire neuron for
    documentation on rest of the behavior.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        vth: ty.Optional[float] = 10,
        refractory_period: ty.Optional[int] = 1,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            vth=vth,
            name=name,
            log_config=log_config,
        )

        if refractory_period < 1:
            raise ValueError("Refractory period must be > 0.")

        self.proc_params["refractory_period"] = refractory_period
        self.refractory_period_end = Var(shape=shape, init=0)
