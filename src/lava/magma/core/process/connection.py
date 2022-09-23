# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from lava.magma.core.learning.learning_rule import LoihiLearningRule
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess

# base class for all connection processes.
from lava.magma.core.process.variable import Var


class ConnectionProcess(AbstractProcess):
    """Base class for connection Processes.

    This base class holds all necessary Vars, Ports and functionality for
    online learning in fixed and floating point simulations. If the
    learning_rule parameter is not set, plasticity is disabled.

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
        shape: tuple = (1, 1),
        learning_rule: LoihiLearningRule = None,
        *args,
        **kwargs,
    ):
        kwargs["learning_rule"] = learning_rule
        kwargs["shape"] = shape

        self.learning_rule = learning_rule

        # Learning Ports
        self.s_in_bap = InPort(shape=(shape[0],))

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

        self.tag_2 = Var(shape=shape, init=0)
        self.tag_1 = Var(shape=shape, init=0)

        super().__init__(*args, **kwargs)
