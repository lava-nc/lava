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
            csp_port: ty.Optional[AbstractCspPort],
            shape: ty.Tuple[int, ...],
            d_type: type):
        self._process_model = process_model
        self._csp_port = csp_port
        self._shape = shape
        self._d_type = d_type

    def start(self):
        if self._csp_port:
            self._csp_port.start()

    def join(self):
        if self._csp_port:
            self._csp_port.join()
