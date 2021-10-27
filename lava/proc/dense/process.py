# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class Dense(AbstractProcess):
    """Dense connections between neurons.
    Realizes the following abstract behavior:
    a_out = W * s_in
    """

    def __init__(self, **kwargs):
        # super(AbstractProcess, self).__init__(kwargs)
        # shape = kwargs.pop("shape")
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, init=kwargs.pop("weights", 0))
