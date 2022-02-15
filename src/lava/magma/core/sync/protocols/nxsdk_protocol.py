# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from dataclasses import dataclass

from lava.magma.core.resources import CPU, NeuroCore
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.runtime_services.runtime_service import (
    PyRuntimeService,
    NxSDKRuntimeService,
)


@dataclass
class NxsdkProtocol(AbstractSyncProtocol):
    """Synchronizer class that implement NxsdkProtocol
    protocol using NxCore for its domain.
    """
    runtime_service = {CPU: PyRuntimeService, NeuroCore: NxSDKRuntimeService}

    @property
    def runtime_service(self):
        return self.runtime_service
