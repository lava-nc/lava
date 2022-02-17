# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from dataclasses import dataclass

from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.runtime_service import AsyncPyRuntimeService


@dataclass
class AsyncProtocol(AbstractSyncProtocol):
    phases = []
    proc_funcions = []

    @property
    def runtime_service(self):
        return {CPU: AsyncPyRuntimeService}
