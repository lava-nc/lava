# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class AbstractLIF(AbstractProcess):
    """Abstract class for variables common to all neurons with leaky
    integrator dynamics."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias = kwargs.pop("bias", 0)
        bias_exp = kwargs.pop("bias_exp", 0)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias = Var(shape=shape, init=bias)
        self.bias_exp = Var(shape=shape, init=bias_exp)


class LIF(AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process.

    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike

    Parameters
    ----------
    du: Inverse of decay time-constant for current decay.
    dv: Inverse of decay time-constant for voltage decay.
    bias: Mantissa part of neuron bias.
    bias_exp: Exponent part of neuron bias, if needed. Mostly for fixed point
              implementations. Ignored for floating point
              implementations.
    vth: Neuron threshold voltage, exceeding which, the neuron will spike.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vth = kwargs.pop("vth", 10)

        self.vth = Var(shape=(1,), init=vth)


class TernaryLIF(AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process with *ternary* spiking
    output, i.e., +1, 0, and -1 spikes. When the voltage of a T-LIF neuron
    exceeds its upper threshold (UTh), it issues a positive spike and when
    the voltage drops below its lower threshold (LTh), it issues a negative
    spike. Between the two thresholds, the neuron follows leaky linear
    dynamics.

    This class inherits the state variables and ports from AbstractLIF and
    adds two new threshold variables for upper and lower thresholds.

    Parameters
    ----------
    vth_hi: Upper threshold voltage, exceeding which the neuron spikes +1
    vth_lo: Lower threshold voltage, below which the neuron spikes -1

    See Also
    --------
    lava.proc.lif.process.LIF: 'Regular' leaky-integrate-and-fire neuron for
    documentation on rest of the parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vth_hi = kwargs.pop("vth_hi", 10)
        vth_lo = kwargs.pop("vth_lo", -10)
        if vth_lo > vth_hi:
            raise AssertionError(f"Lower threshold {vth_lo} is larger than the "
                                 f"upper threshold {vth_hi} for Ternary LIF "
                                 f"neurons. Consider switching the values.")
        self.vth_hi = Var(shape=(1,), init=vth_hi)
        self.vth_lo = Var(shape=(1,), init=vth_lo)
