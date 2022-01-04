# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

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


class ResetVar(AbstractProcess):
    """Resets the variable of another process linked to this object at a
    set interval and offset (phase).

    Parameters
    ----------
    reset_var : Var
        the variable that needs to be reset.
    reset_value : int or float
        reset value, by default 0
    interval : int, optional
        reset interval, by default 1
    offset : int, optional
        reset offset (phase), by default 0
    """
    def __init__(
        self,
        reset_var: Var,
        reset_value: Union[int, float] = 0,
        interval: int = 1,
        offset: int = 0,
    ) -> None:
        super().__init__()
        self.reset_value = Var((1,), reset_value)
        self.interval = Var((1,), interval)
        self.offset = Var((1,), offset % interval)
        self.state = RefPort(reset_var.shape)
        self.state.connect_var(reset_var)


class AbstractPyResetVar(PyLoihiProcessModel):
    """Abstract Reset process implementation."""
    state = None
    reset_value: np.ndarray = LavaPyType(np.ndarray, int)
    interval: np.ndarray = LavaPyType(np.ndarray, int)
    offset: np.ndarray = LavaPyType(np.ndarray, int)

    def post_guard(self) -> None:
        return self.current_ts % self.interval == self.offset

    def run_post_mgmt(self) -> None:
        self.state.write(0 * self.state.read() + self.reset_value)


@implements(proc=ResetVar, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyResetVarFixed(AbstractPyResetVar):
    """Reset process implementation for int type."""
    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)


@implements(proc=ResetVar, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyResetVarFloat(AbstractPyResetVar):
    """Reset process implementation for float type."""
    state: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
