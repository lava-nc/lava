# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing
import numpy as np
from enum import IntEnum, unique

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort



@unique
class ACTIVATION_MODE(IntEnum):
    """Enum for synapse sigma delta activation mode. Options are {``ReLU : 0``}.
    """
    Unit = 0
    ReLU = 1


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
    """
    '''

    # cumulative error

    '''
    def __init__(
        self,
        **kwargs: typing.Union[int, typing.Tuple[int, ...]]
    ) -> None:
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (1,))
        act_mode = kwargs.get('act_mode', ACTIVATION_MODE.ReLU)
        vth = kwargs.get('vth')
        cum_error = kwargs.get('cum_error', False)
        bias = kwargs.pop('bias', 0)
        wgt_exp = kwargs.pop('wgt_exp', 6)
        # scaling factor for fixed precision scaling
        state_exp = kwargs.pop('state_exp', 6)
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
