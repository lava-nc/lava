# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod

import typing as ty

from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.runtime_services.runtime_service import \
    AbstractRuntimeService


class AbstractProcessBuilder(ABC):
    """An AbstractProcessBuilder is the base type for process builders.

    Process builders instantiate and initialize a ProcessModel.

    Parameters
        ----------
        proc_model: AbstractProcessModel
                    ProcessModel class of the process to build.
        model_id: int
                  model_id represents the ProcessModel ID to build.
    """
    def __init__(
            self,
            proc_model: ty.Type[AbstractProcessModel],
            model_id: int):
        self._proc_model = proc_model
        self._model_id = model_id

    @property
    @abstractmethod
    def proc_model(self) -> "AbstractProcessModel":
        pass


class AbstractRuntimeServiceBuilder(ABC):
    """An AbstractRuntimeServiceBuilder is the base type for
    RuntimeService builders.

    RuntimeService builders instantiate and initialize a RuntimeService.

    Parameters
        ----------
        rs_class: AbstractProcessModel
                  ProcessModel class of the process to build.
        sync_protocol: AbstractSyncProtocol
                       Synchronizer class that implements a protocol
                       in a domain.
    """
    def __init__(self,
                 rs_class: ty.Type[AbstractRuntimeService],
                 sync_protocol: ty.Type[AbstractSyncProtocol]):
        self.rs_class = rs_class
        self.sync_protocol = sync_protocol

    @property
    @abstractmethod
    def runtime_service_id(self):
        pass

    def build(self):
        raise NotImplementedError(
            "build function for RuntimeServiceBuilder is not implemented"
        )


class AbstractChannelBuilder(ABC):
    """An AbstractChannelBuilder is the base type for
    channel builders which build communication channels
    between services and processes"""

    pass
