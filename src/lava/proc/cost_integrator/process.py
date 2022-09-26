# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class CostIntegrator(AbstractProcess):
    """Node that integrates cost components and produces output when a better
    cost is found.

    Parameters
    ----------
    shape : tuple(int)
        The expected number and topology of the input cost components.
    name : str, optional
        Name of the Process. Default is 'Process_ID', where ID is an
        integer value that is determined automatically.
        log_config: Configuration options for logging.

    InPorts
    -------
    cost_in
        input to be additively integrated.

    OutPorts
    --------
    update_buffer
        OutPort which notifies the next process about the
        detection of a better cost.

    Vars
    ----
    cost
        Holds current cost as addition of input spikes' payloads

    min_cost
        Current minimum cost, i.e., the lowest reported cost so far.
    """

    def __init__(self, *, shape: ty.Tuple[int, ...] = (1,),
                 min_cost: int = 2**24,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(shape=shape, name=name, log_config=log_config)
        self.cost_in = InPort(shape=shape)
        self.update_buffer = OutPort(shape=shape)
        self.cost = Var(shape=shape, init=2**24)
        self.min_cost = Var(shape=shape, init=min_cost)
