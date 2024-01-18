# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from enum import IntEnum, unique

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


@unique
class ActivationMode(IntEnum):
    """Enum for synapse sigma delta activation mode. Options are
    UNIT: 0
    RELU: 1
    """
    UNIT = 0
    RELU = 1


class Sigma(AbstractProcess):
    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...]) -> None:
        """Sigma integration unit process definition. A sigma process is simply
        a cumulative accumulator over time.

        Sigma dynamics:
        sigma = a_in + sigma                      # sigma dendrite
        a_out = sigma

        Parameters
        ----------
        shape: Tuple
            shape of the sigma process. Default is (1,).
        """
        super().__init__(shape=shape)

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.sigma = Var(shape=shape, init=0)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']


class Delta(AbstractProcess):
    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...],
                 vth: ty.Union[int, float],
                 cum_error: ty.Optional[bool] = False,
                 spike_exp: ty.Optional[int] = 0,
                 state_exp: ty.Optional[int] = 0) -> None:
        """Delta process definition. Spike mechanism based on accumulated error
        is also supported.

        Delta dynamics:
        delta   = act_new - act + residue           # delta encoding
        s_out   = delta if abs(delta) > vth else 0  # spike mechanism
        residue = delta - s_out                     # residue accumulation
        act     = act_new

        Delta dynamics (with cumulative error):
        delta   = act_new - act + residue           # delta encoding
        error   = error + delta                     # error accumulation
        s_out   = delta if abs(error) > vth else 0  # spike mechanism
        error   = error * (1 - H(s_out))            # error reset
        residue = delta - s_out                     # residue accumulation
        act = act

        Parameters
        ----------
        shape: Tuple
            Shape of the sigma process.
        vth: int or float
            Threshold of the delta encoder.
        cum_error: Bool
            Flag to enable/disable cumulative error accumulation.
            Default is False.
        spike_exp: int
            Scaling exponent with base 2 for the spike message.
            Note: This should only be used for fixed point models.
            Default is 0.
        state_exp: int
            Scaling exponent with base 2 for the state variables.
            Note: This should only be used for fixed point models.
            Default is 0.
        """
        super().__init__(shape=shape, vth=vth, cum_error=cum_error,
                         spike_exp=spike_exp, state_exp=state_exp)

        vth = vth * (1 << (spike_exp + state_exp))

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.vth = Var(shape=(1,), init=vth)
        self.sigma = Var(shape=shape, init=0)
        self.act = Var(shape=shape, init=0)
        self.residue = Var(shape=shape, init=0)
        self.error = Var(shape=shape, init=0)
        self.spike_exp = Var(shape=(1,), init=spike_exp)
        self.state_exp = Var(shape=(1,), init=state_exp)
        self.cum_error = Var(shape=(1,), init=cum_error)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']


class SigmaDelta(AbstractProcess):
    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            vth: float,
            bias: ty.Optional[float] = 0,
            act_mode: ty.Optional[ActivationMode] = ActivationMode.RELU,
            cum_error: ty.Optional[bool] = False,
            spike_exp: ty.Optional[int] = 0,
            state_exp: ty.Optional[int] = 0,
            **kwargs) -> None:
        """Sigma delta neuron process. At the moment only ReLu activation is
        supported. Spike mechanism based on accumulated error is also supported.

        Sigma Delta dynamics:
        sigma   = a_in + sigma                      # sigma dendrite
        act_new = act_fx(sigma + bias)              # activation
        delta   = act_new - act + residue           # delta encoding
        s_out   = delta if abs(delta) > vth else 0  # spike mechanism
        residue = delta - s_out                     # residue accumulation
        act     = act_new

        Sigma Delta dynamics (with cumulative error):
        sigma   = a_in + sigma                      # sigma dendrite
        act_new = act_fx(sigma + bias)              # activation
        delta   = act_new - act + residue           # delta encoding
        error   = error + delta                     # error accumulation
        s_out   = delta if abs(error) > vth else 0  # spike mechanism
        error   = error * (1 - H(s_out))            # error reset
        residue = delta - s_out                     # residue accumulation
        act = act

        Parameters
        ----------
        shape: Tuple
            shape of the sigma process. Default is (1,).
        vth: int or float
            threshold of the delta encoder.
        bias: int or float
            bias to the neuron activation.
        act_mode: enum
            activation mode describing the non-linear activation function.
            Options are described by ``ActivationMode`` enum.
        cum_error: Bool
            flag to enable/disable cumulative error accumulation.
            Default is False.
        spike_exp: int
            Scaling exponent with base 2 for the spike message.
            Note: This should only be used for fixed point models.
            Default is 0.
        state_exp: int
            Scaling exponent with base 2 for the state variables.
            Note: This should only be used for fixed point models.
            Default is 0.
        """
        super().__init__(shape=shape, vth=vth, bias=bias,
                         act_mode=act_mode, cum_error=cum_error,
                         spike_exp=spike_exp, state_exp=state_exp, **kwargs)
        # scaling factor for fixed precision scaling
        vth = vth * (1 << (spike_exp + state_exp))
        bias = bias * (1 << (spike_exp + state_exp))

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.vth = Var(shape=(1,), init=vth)
        self.sigma = Var(shape=shape, init=0)
        self.act = Var(shape=shape, init=0)
        self.residue = Var(shape=shape, init=0)
        self.error = Var(shape=shape, init=0)
        self.bias = Var(shape=shape, init=bias)
        self.spike_exp = Var(shape=(1,), init=spike_exp)
        self.state_exp = Var(shape=(1,), init=state_exp)
        self.cum_error = Var(shape=(1,), init=cum_error)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']
