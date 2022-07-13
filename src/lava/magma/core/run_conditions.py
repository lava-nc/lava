# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC


class AbstractRunCondition(ABC):
    """Base class for run conditions.

    RunConditions specify for how long a process will run.

    Parameters
    ----------
    blocking: bool
        If set to True, blocks further commands from execution until returns.
    """

    def __init__(self, blocking: bool):
        self.blocking = blocking


class RunSteps(AbstractRunCondition):
    """Runs a process for a specified number of time steps with respect to a
    SyncDomain assigned to any sub processes.

    Parameters
    ----------
    num_steps: int
        Number of steps to be run with respect to the SyncDomain.
    blocking: bool
        If set to True, blocks further commands from execution until returns.
        (Default = True)
    """

    def __init__(self, num_steps: int, blocking: bool = True):
        super().__init__(blocking)
        self.num_steps = num_steps


class RunContinuous(AbstractRunCondition):
    """Runs a Process continuously without a time step limit (non-blocking).

    Using this RunCondition, the runtime runs continuously and non-blocking.
    This means that the runtime must be paused or stopped manually by calling
    `pause()` or `stop()` from the running process.
    The runtime can be continued after `pause()` by calling `run()` again.
    """

    def __init__(self):
        super().__init__(blocking=False)
