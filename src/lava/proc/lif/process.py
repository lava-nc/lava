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
from lava.proc.io.utils import convert_to_numpy_array


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


class AbstractEILIF(AbstractProcess):
    """Abstract class for variables common to all neurons with Excitatory/Inhibitory 
    leaky integrator dynamics and configurable time constants"""

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u_exc: ty.Union[float, list, np.ndarray],
        u_inh: ty.Union[float, list, np.ndarray],
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du_exc: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du_inh: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        dv: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_mant: ty.Union[float, list, np.ndarray],
        bias_exp: ty.Union[float, list, np.ndarray],
        name: str,
        log_config: LogConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u_exc=u_exc,
            u_inh=u_inh,
            v=v,
            du_exc=du_exc,
            du_inh=du_inh,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u_exc = Var(shape=shape, init=u_exc)
        self.u_inh = Var(shape=shape, init=u_inh)
        self.u = Var(shape=shape, init=u_exc + u_inh)     # neuron total current (u_inh is negative)
        self.v = Var(shape=shape, init=v)
        self.du_exc = Var(shape=shape, init=du_exc)     # Shape of du_exc must match the shape of the neurons
        self.du_inh = Var(shape=shape, init=du_inh)     # Shape of du_inh must match the shape of the neurons
        self.dv = Var(shape=shape, init=dv)     # Shape of dv must match the shape of the neurons
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.bias_mant = Var(shape=shape, init=bias_mant)


class EILIF(AbstractEILIF):
    """Exctitatory/Inhibitory Leaky-Integrate-and-Fire (LIF) neural Process.
    This neuron model receives 2 input currents, one excitatory and one inhibitory.
    The neuron's total current is the sum of the excitatory and inhibitory currents.
    Each current has its own decay time-constant and it is independent on a neuron-to-neuron basis.

    LIF dynamics abstracts to:
    u_exc[t] = u_exc[t-1] * (1-du_exc) + a_in (excitatory spike)         # neuron excitatory current
    u_inh[t] = u_inh[t-1] * (1-du_inh) + a_in (inhibitory spike)        # neuron inhibitory current

    u[t] = u_exc[t] + u_inh[t]                               # neuron total current (u_inh[t] is negative)
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u_exc : float, list, numpy.ndarray, optional
        Initial value of the neurons' excitatory current.
    u_inh : float, list, numpy.ndarray, optional
        Initial value of the neurons' inhibitory current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du_exc : float, list, numpy.ndarray, optional
        Inverse of decay time-constant for excitatory current decay. This can be a scalar, list,
        or numpy array. Anyhow, it will be converted to a np array representing the 
        time-constants of each neuron.
    du_inh : float, list, numpy.ndarray, optional
        Inverse of decay time-constant for inhibitory current decay. This can be a scalar, list,
        or numpy array. Anyhow, it will be converted to a np array representing the 
        time-constants of each neuron.
    dv : float, list, numpy.ndarray, optional
        Inverse of decay time-constant for voltage decay. This can be a scalar, list,
        or numpy array. Anyhow, it will be converted to a np array representing the 
        time-constants of each neuron.
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
    >>> ei_lif = EILIF(shape=(200, 15), du_exc=0.1, du_inh=0.2, dv=5)
    This will create 200x15 EILIF neurons that all have the same excitatory and 
    inhibitory current decays (0.1 and 0.2, respectively) and voltage decay of 5.
    """
    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            u_exc: ty.Union[float, list, np.ndarray] = 0,
            u_inh: ty.Union[float, list, np.ndarray] = 0,
            du_exc: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            du_inh: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            dv: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            vth: ty.Optional[float] = 10,
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None,
            verbose: ty.Optional[bool] = False,
            **kwargs,
    ) -> None:
        # Try to convert du_exc, du_inh and dv to numpy arrays if they are not already
        # If unsuccessful, it will raise a ValueError
        du_exc = convert_to_numpy_array(du_exc, shape, "du_exc", verbose=verbose)
        du_inh = convert_to_numpy_array(du_inh, shape, "du_inh", verbose=verbose)
        dv = convert_to_numpy_array(dv, shape, "dv", verbose=verbose)
        
        super().__init__(
            shape=shape,
            u_exc=u_exc,
            u_inh=u_inh,
            v=v,
            du_exc=du_exc,
            du_inh=du_inh,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            vth=vth,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        # Add the vth variable to the process
        self.vth = Var(shape=(1,), init=vth)

class EILIFRefractory(EILIF):
    """Excitatory/Inhibitory Leaky-Integrate-and-Fire (LIF) neural Process with refractory period.
    This neuron model receives 2 input currents, one excitatory and one inhibitory.
    The neuron's total current is the sum of the excitatory and inhibitory currents.
    Each current has its own decay time-constant and it is independent on a neuron-to-neuron basis.

    LIF dynamics abstracts to:
    u_exc[t] = u_exc[t-1] * (1-du_exc) + a_in (excitatory spike)         # neuron excitatory current
    u_inh[t] = u_inh[t-1] * (1-du_inh) + a_in (inhibitory spike)        # neuron inhibitory current

    u[t] = u_exc[t] + u_inh[t]                               # neuron total current (u_inh[t] is negative)
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u_exc : float, list, numpy.ndarray, optional
        Initial value of the neurons' excitatory current.
    u_inh : float, list, numpy.ndarray, optional
        Initial value of the neurons' inhibitory current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du_exc : float, list, numpy.ndarray, optional
        Inverse of decay time-constant for excitatory current decay. This can be a scalar, list,
        or numpy array. Anyhow, it will be converted to a np array representing the 
        time-constants of each neuron.
    du_inh : float, list, numpy.ndarray, optional
        Inverse of decay time-constant for inhibitory current decay. This can be a scalar, list,
        or numpy array. Anyhow, it will be converted to a np array representing the 
        time-constants of each neuron.
    dv : float, list, numpy.ndarray, optional
        Inverse of decay time-constant for voltage decay. This can be a scalar, list,
        or numpy array. Anyhow, it will be converted to a np array representing the 
        time-constants of each neuron.
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

    Example
    -------
    >>> refrac_ei_lif = EILIFRefractory(shape=(200, 15), du_exc=0.1, du_inh=0.2, dv=5)
    This will create 200x15 EILIF neurons that all have the same excitatory and 
    inhibitory current decays (0.1 and 0.2, respectively), voltage decay of 5.
    and refractory period of 1 timestep.
    """
    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            u_exc: ty.Union[float, list, np.ndarray] = 0,
            u_inh: ty.Union[float, list, np.ndarray] = 0,
            du_exc: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            du_inh: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            dv: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
            vth: ty.Optional[float] = 10,
            refractory_period: ty.Optional[int] = 1,
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None,
            verbose: ty.Optional[bool] = False,
            **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            u_exc=u_exc,
            u_inh=u_inh,
            v=v,
            du_exc=du_exc,
            du_inh=du_inh,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            vth=vth,
            name=name,
            log_config=log_config,
            verbose=verbose,
            **kwargs,
        )

        # Validate the refractory period
        if refractory_period < 1:   # TODO: Change to 0
            raise ValueError("Refractory period must be > 0.")
        # Check if the refractory period is a float
        if isinstance(refractory_period, float):
            if verbose:
                print("Refractory period must be an integer. Converting to integer...")
            refractory_period = int(refractory_period)

        self.proc_params["refractory_period"] = refractory_period
        self.refractory_period_end = Var(shape=shape, init=0)
