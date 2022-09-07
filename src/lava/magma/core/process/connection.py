# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.learning.learning_rule import LearningRule

# base class for all connection processes.
from lava.magma.core.process.variable import Var


class ConnectionProcess(AbstractProcess):
    def __init__(
        self,
        shape: tuple = (1, 1),
        learning_rule: LearningRule = None,
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
