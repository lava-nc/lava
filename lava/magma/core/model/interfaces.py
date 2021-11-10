# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC

from lava.magma.compiler.channels.interfaces import AbstractCspPort


# ToDo: (AW) This type hierarchy is still not clean. csp_port could be a
#  CspSendPort or CspRecvPort so down-stream classes can't do proper type
#  inference to determine if there is a send/peek/recv/probe method.
class AbstractPortImplementation(ABC):
    def __init__(
        self,
        process_model: "AbstractProcessModel",  # noqa: F821
        csp_ports: ty.List[AbstractCspPort] = [],
        shape: ty.Tuple[int, ...] = tuple(),
        d_type: type = int,
    ):
        self._process_model = process_model
        self._csp_ports = (
            csp_ports if isinstance(csp_ports, list) else [csp_ports]
        )
        self._shape = shape
        self._d_type = d_type

    def start(self):
        for csp_port in self._csp_ports:
            csp_port.start()

    def join(self):
        for csp_port in self._csp_ports:
            csp_port.join()
