# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing
from enum import IntEnum, unique

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


@unique
class ACTIVATION_MODE(IntEnum):
    """Enum for synapse sigma delta activation mode. Options are
    {``Unit : 0``, ``ReLU : 1``}.
    """
    Unit = 0
    ReLU = 1


class Sigma(AbstractProcess):
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
    def __init__(
        self,
        **kwargs: typing.Union[int, typing.Tuple[int, ...]]
    ) -> None:
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (1,))

        self.shape = shape

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.sigma = Var(shape=shape, init=0)


class Delta(AbstractProcess):
    """Delta process definition. Spike mechanism based on accumulated error
    is also supported.

    Delta dynamics:
    delta   = act_new - act + residue           # delta encoding
    s_out   = delta if abs(delta) > vth else 0  # spike mechanism
    resiude = delta - s_out                     # residue accumulation
    act     = act_new

    Delta dynamics (with cumulative error):
    delta   = act_new - act + residue           # delta encoding
    error   = error + delta                     # error accumulation
    s_out   = delta if abs(error) > vth else 0  # spike mechanism
    error   = error * (1 - H(s_out))            # error reset
    resiude = delta - s_out                     # residue accumulation
    act = act

    Parameters
    ----------
    shape: Tuple
        shape of the sigma process. Default is (1,).
    vth: int or float
        threshold of the delta encoder.
    cum_error: Bool
        flag to enable/disable cumulative error accumulation. Default is False.
    wgt_exp: int
        weight scaling exponent. Note: this has effect only on fixed point
        models. Default is 0.
    state_exp: int
        state variables scaling exponent. Note: this has effect only on fixed
        point modles. Default is 0.
    """
    def __init__(
        self,
        **kwargs: typing.Union[int, typing.Tuple[int, ...]]
    ) -> None:
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (1,))
        cum_error = kwargs.get('cum_error', False)
        wgt_exp = kwargs.pop('wgt_exp', 0)
        # scaling factor for fixed precision scaling
        state_exp = kwargs.pop('state_exp', 0)
        vth = kwargs.get('vth') * (1 << (wgt_exp + state_exp))

        self.shape = shape

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.vth = Var(shape=(1,), init=vth)
        self.sigma = Var(shape=shape, init=0)
        self.act = Var(shape=shape, init=0)
        self.residue = Var(shape=shape, init=0)
        self.error = Var(shape=shape, init=0)
        self.wgt_exp = Var(shape=(1,), init=wgt_exp)
        self.state_exp = Var(shape=(1,), init=state_exp)
        self.cum_error = Var(shape=(1,), init=cum_error)


class SigmaDelta(AbstractProcess):
    """Sigma delta neuron process. At the moment only ReLu activation is
    supported. Spike mechanism based on accumulated error is also supported.

    Sigma Delta dynamics:
    sigma   = a_in + sigma                      # sigma dendrite
    act_new = act_fx(sigma + bias)              # activation
    delta   = act_new - act + residue           # delta encoding
    s_out   = delta if abs(delta) > vth else 0  # spike mechanism
    resiude = delta - s_out                     # residue accumulation
    act     = act_new

    Sigma Delta dynamics (with cumulative error):
    sigma   = a_in + sigma                      # sigma dendrite
    act_new = act_fx(sigma + bias)              # activation
    delta   = act_new - act + residue           # delta encoding
    error   = error + delta                     # error accumulation
    s_out   = delta if abs(error) > vth else 0  # spike mechanism
    error   = error * (1 - H(s_out))            # error reset
    resiude = delta - s_out                     # residue accumulation
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
        activation mode describing the non-linear activation function. Options
        are described by ``ACTIVATION_MODE`` enum.
    cum_error: Bool
        flag to enable/disable cumulative error accumulation. Default is False.
    wgt_exp: int
        weight scaling exponent. Note: this has effect only on fixed point
        models. Default is 0.
    state_exp: int
        state variables scaling exponent. Note: this has effect only on fixed
        point modles. Default is 0.
    """
    def __init__(
        self,
        **kwargs: typing.Union[int, typing.Tuple[int, ...]]
    ) -> None:
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (1,))
        act_mode = kwargs.get('act_mode', ACTIVATION_MODE.ReLU)
        cum_error = kwargs.get('cum_error', False)
        wgt_exp = kwargs.pop('wgt_exp', 0)
        # scaling factor for fixed precision scaling
        state_exp = kwargs.pop('state_exp', 0)
        vth = kwargs.get('vth') * (1 << (wgt_exp + state_exp))
        bias = kwargs.pop('bias', 0) * (1 << (wgt_exp + state_exp))
        self.proc_params['act_fn'] = act_mode

        self.shape = shape

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.vth = Var(shape=(1,), init=vth)
        self.sigma = Var(shape=shape, init=0)
        self.act = Var(shape=shape, init=0)
        self.residue = Var(shape=shape, init=0)
        self.error = Var(shape=shape, init=0)
        self.bias = Var(shape=shape, init=bias)
        self.wgt_exp = Var(shape=(1,), init=wgt_exp)
        self.state_exp = Var(shape=(1,), init=state_exp)
        self.cum_error = Var(shape=(1,), init=cum_error)
