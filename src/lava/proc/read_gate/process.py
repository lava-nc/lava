# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class ReadGate(AbstractProcess):
    """Process that triggers solution readout when problem is solved.

    Parameters
    ----------
    shape: The shape of the set of units in the downstream process whose state
        will be read by ReadGate.
    target_cost: cost value at which, once attained by the network,
        this process will stop execution.
    name: Name of the Process. Default is 'Process_ID', where ID is an
        integer value that is determined automatically.
    log_config: Configuration options for logging.

    InPorts
    -------
    cost_in: Receives a better cost found by the CostIntegrator at the
        previous timestep.

    OutPorts
    --------
    cost_out: Forwards to an upstream process the better cost notified by the
        CostIntegrator.
    solution_out: Forwards to an upstream process the better variable assignment
        found by the solver network.
    send_pause_request: Notifies upstream process to request execution to pause.
    """

    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 target_cost=None,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(shape=shape,
                         target_cost=target_cost,
                         name=name,
                         log_config=log_config)
        self.target_cost = Var(shape=(1,), init=target_cost)
        self.cost_in = InPort(shape=(1,))
        self.acknowledgemet = InPort(shape=(1,))
        self.cost_out = OutPort(shape=(1,))
        self.send_pause_request = OutPort(shape=(1,))
        self.solution_out = OutPort(shape=shape)
        self.solution_reader = RefPort(shape=shape)
