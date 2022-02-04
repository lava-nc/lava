# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABC, abstractmethod

import typing as ty

from lava.magma.compiler.channels.interfaces import AbstractCspPort
if ty.TYPE_CHECKING:
    from lava.magma.core.model.model import AbstractProcessModel


class AbstractProcessBuilder(ABC):
    @abstractmethod
    def set_csp_ports(self, csp_ports: ty.List[AbstractCspPort]):
        pass

    @property
    @abstractmethod
    def proc_model(self) -> "AbstractProcessModel":
        pass


class AbstractRuntimeServiceBuilder(ABC):
    def __init__(self, rs_class, sync_protocol):
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
