# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.learning.learning_rule import LoihiLearningRule
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess

# base class for all connection processes.
from lava.magma.core.process.variable import Var


class LearningConnectionProcess(AbstractProcess):
    """Base class for connection Processes.

    This base class holds all necessary Vars, Ports and functionality for
    online learning in fixed and floating point simulations.

    Attributes
    ----------
    s_in_bap: InPort
        Input port to receive back-propagating action potentials (BAP)
    x0: Var
        Conditional for pre-synaptic spike times (is 1 if pre-synaptic neurons
        spiked in this time-step).
    tx: Var
        Within-epoch spike times of pre-synaptic neurons.
    x1: Var
        First pre-synaptic trace.
    x2: Var
        Second pre-synaptic trace.
    y0: Var
        Conditional for post-synaptic spike times (is 1 if post-synaptic neurons
        spiked in this time-step).
    ty: Var
        Within-epoch spike times of post-synaptic neurons.
    y1: Var
        First post-synaptic trace.
    y2: Var
        Second post-synaptic trace.
    y3: Var
        Third post-synaptic trace.
    tag_1: Var
        Tag synaptic variable
    tag_2: Var
        Delay synaptic variable

    Parameters
    ----------
    shape: tuple, ndarray
        Shape of the connection in format (post, pre) order.
    learning_rule: LoihiLearningRule
        Learning rule which determines the parameters for online learning.
    """

    def __init__(
        self,
        shape: tuple,
        learning_rule: ty.Optional[LoihiLearningRule],
        **kwargs,
    ):
        kwargs["learning_rule"] = learning_rule
        kwargs["shape"] = shape
        tag_1 = kwargs.get("tag_1", 0)
        tag_2 = kwargs.get("tag_2", 0)

        self._learning_rule = learning_rule

        # Learning Ports
        self.s_in_bap = InPort(shape=(shape[0],))
        self.s_in_y1 = InPort(shape=(shape[0],))
        self.s_in_y2 = InPort(shape=(shape[0],))
        self.s_in_y3 = InPort(shape=(shape[0],))

        # Learning Vars
        self.x0 = Var(shape=(shape[-1],), init=0)
        self.tx = Var(shape=(shape[-1],), init=0)
        self.x1 = Var(shape=(shape[-1],), init=0)
        self.x2 = Var(shape=(shape[-1],), init=0)

        self.y0 = Var(shape=(shape[0],), init=0)
        self.ty = Var(shape=(shape[0],), init=0)
        self.y1 = Var(shape=(shape[0],), init=0)
        self.y2 = Var(shape=(shape[0],), init=0)
        self.y3 = Var(shape=(shape[0],), init=0)

        self.tag_1 = Var(shape=shape, init=tag_1)
        self.tag_2 = Var(shape=shape, init=tag_2)

        self.dw = Var(shape=(256,), init=learning_rule.dw_str)
        self.dd = Var(shape=(256,), init=learning_rule.dd_str)
        self.dt = Var(shape=(256,), init=learning_rule.dt_str)

        self.x1_tau = Var(shape=(1,), init=learning_rule.x1_tau)
        self.x1_impulse = Var(shape=(1,), init=learning_rule.x1_impulse)
        self.x2_tau = Var(shape=(1,), init=learning_rule.x2_tau)
        self.x2_impulse = Var(shape=(1,), init=learning_rule.x2_impulse)

        self.y1_tau = Var(shape=(1,), init=learning_rule.y1_tau)
        self.y1_impulse = Var(shape=(1,), init=learning_rule.y1_impulse)
        self.y2_tau = Var(shape=(1,), init=learning_rule.y2_tau)
        self.y2_impulse = Var(shape=(1,), init=learning_rule.y2_impulse)
        self.y3_tau = Var(shape=(1,), init=learning_rule.y3_tau)
        self.y3_impulse = Var(shape=(1,), init=learning_rule.y3_impulse)

        super().__init__(**kwargs)
