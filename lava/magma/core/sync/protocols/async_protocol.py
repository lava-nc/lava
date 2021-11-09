# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from dataclasses import dataclass
import typing as ty

from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.synchronizer import LoihiSynchronizer


@dataclass
class AsyncProtocol(AbstractSyncProtocol):
    phases = []
    proc_funcions = []

    @property
    def synchronizer(self) -> ty.Dict[ty.Type, ty.Type]:
        return {CPU: LoihiSynchronizer}
