# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod

from lava.magma.compiler.channels.pypychannel import (
    CspRecvPort,
    CspSendPort
)
from lava.magma.core.sync.protocol import AbstractSyncProtocol


class AbstractRuntimeService(ABC):
    def __init__(self, protocol):
        self.protocol: ty.Optional[AbstractSyncProtocol] = protocol

        self.runtime_service_id: ty.Optional[int] = None

        self.runtime_to_service: ty.Optional[CspRecvPort] = None
        self.service_to_runtime: ty.Optional[CspSendPort] = None

        self.model_ids: ty.List[int] = []

    def __repr__(self):
        return f"Synchronizer : {self.__class__}, \
                 RuntimeServiceId : {self.runtime_service_id}, \
                 Protocol: {self.protocol}"

    def start(self):
        self.runtime_to_service.start()
        self.service_to_runtime.start()
        self.run()

    @abstractmethod
    def run(self):
        pass

    def join(self):
        self.runtime_to_service.join()
        self.service_to_runtime.join()
