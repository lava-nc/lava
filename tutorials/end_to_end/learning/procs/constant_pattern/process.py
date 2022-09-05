# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import OutPort


from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class ConstantPattern(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.pop("shape")
        init_value = kwargs.pop("init_value", 0)

        self.null_pattern = Var(shape=shape, init=np.full(shape, np.nan))
        self._value = Var(shape=(1,), init=init_value)

        self.changed = Var(shape=(1,), init=np.array([True]))

        self.a_out = OutPort(shape=shape)

    def _update(self):
        self.changed.set(np.array([True]))

        self.changed.get()

    @property
    def value(self):
        try:
            return self._value.get()
        except AttributeError:
            return None

    @value.setter
    def value(self, value):
        self._value.set(value)

        self._value.get()

        self._update()


@implements(proc=ConstantPattern, protocol=LoihiProtocol)
@requires(CPU)
class ConstantPatternProcessModel(PyLoihiProcessModel):
    null_pattern: np.ndarray = LavaPyType(np.ndarray, float)
    _value: np.ndarray = LavaPyType(np.ndarray, float)

    changed: np.ndarray = LavaPyType(np.ndarray, bool)

    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        if self.changed[0]:
            pattern = np.full_like(self.null_pattern, self._value[0])

            self.changed[0] = False

            self.a_out.send(pattern)
        else:
            self.a_out.send(self.null_pattern)