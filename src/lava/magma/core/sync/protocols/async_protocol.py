# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from dataclasses import dataclass

from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.runtime_services.runtime_service import \
    AsyncPyRuntimeService


@dataclass
class AsyncProtocol(AbstractSyncProtocol):
    """
    Protocol to run processes asynchronously.

    With the AsyncProtocol, Processes are executed without synchronization
    on specific phases. This means that the Processes could run with
    a varying speed and message passing is possible at any time.

    `AsyncProtocol` is currently only implemented for execution on CPU using
    the `PyAsyncProcessModel` in which the `run_async()` function must be
    implemented to define the behavior of the underlying `Process`.

    For example:

    >>>    @implements(proc=SimpleProcess, protocol=AsyncProtocol)
    >>>    @requires(CPU)
    >>>    class SimpleProcessModel(PyAsyncProcessModel):
    >>>        u = LavaPyType(int, int)
    >>>        v = LavaPyType(int, int)

    >>>        def run_async(self):
    >>>            while True:
    >>>                self.u = self.u + 10
    >>>                self.v = self.v + 1000
    >>>                if self.check_for_stop_cmd():
    >>>                    return
    """

    phases = []
    proc_functions = []

    @property
    def runtime_service(self):
        return {CPU: AsyncPyRuntimeService}
