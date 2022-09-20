# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from typing import Union

from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.process.process import AbstractProcess

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyRefPort


class Reset(AbstractProcess):
    """Resets it's internal state at a set interval and offset (phase).

    Parameters
    ----------
    reset_value : int or float
        reset value, by default 0
    interval : int, optional
        reset interval, by default 1
    offset : int, optional
        reset offset (phase), by default 0
    """
    def __init__(
        self,
        *,
        reset_value: Union[int, float] = 0,
        interval: int = 1,
        offset: int = 0,
    ) -> None:
        super().__init__(reset_value=reset_value, interval=interval,
                         offset=offset)
        self.reset_value = Var((1,), init=reset_value)
        self.interval = Var((1,), init=interval)
        self.offset = Var((1,), init=offset % interval)
        self.state = RefPort((1,))

    def connect_var(self, var: Var) -> None:
        self.state = RefPort(var.shape)
        self.state.connect_var(var)
        self._post_init()


class AbstractPyReset(PyLoihiProcessModel):
    """Abstract Reset process implementation."""
    state: Union[PyRefPort, None] = None
    reset_value: np.ndarray = LavaPyType(np.ndarray, int)
    interval: np.ndarray = LavaPyType(np.ndarray, int)
    offset: np.ndarray = LavaPyType(np.ndarray, int)

    def post_guard(self) -> None:
        return (self.time_step - 1) % self.interval == self.offset

    def run_post_mgmt(self) -> None:
        self.state.write(np.zeros(self.state._shape,
                                  self.state._d_type) + self.reset_value)
        self.state.wait()  # ensures write() has finished before moving on


@implements(proc=Reset, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyResetFixed(AbstractPyReset):
    """Reset process implementation for int type."""
    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)


@implements(proc=Reset, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyResetFloat(AbstractPyReset):
    """Reset process implementation for float type."""
    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
