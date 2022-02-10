# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from abc import ABC


class AbstractRunCondition(ABC):
    """Specifies run condition until when a process is run."""

    def __init__(self, blocking):
        self.blocking = blocking


class RunSteps(AbstractRunCondition):
    """Runs a process for a specified number of time steps with respect to a
    SyncDomain assigned to any sub processes."""

    def __init__(self, num_steps: int, blocking: bool = True):
        super().__init__(blocking)
        self.num_steps = num_steps


class RunContinuous(AbstractRunCondition):
    """Runs a Process continuously without a time step limit."""

    def __init__(self):
        super().__init__(blocking=False)
