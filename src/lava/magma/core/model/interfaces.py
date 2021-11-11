# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC
from enum import IntEnum, unique
import numpy as np

from lava.magma.compiler.channels.interfaces import (
    AbstractCspRecvPort,
    AbstractCspSendPort)

from lava.magma.core.model.model import AbstractProcessModel


@unique
class PortMessageFormat(IntEnum):
    VECTOR_DENSE = 1
    VECTOR_SPARSE = 2
    SCALAR_DENSE = 3
    SCALAR_SPARSE = 4
    DONE = 1111


class AbstractPortMessageHeader(ABC):

    def __init__(self,
                 format: 'PortMessageFormat',
                 number_elements: ty.Type[int]):
        self._format = format
        self._number_elements = number_elements

    def return_header(self) -> ty.Tuple['PortMessageFormat', ty.Type[int]]:
        return self._format, self._number_elements


class AbstractPortMessagePayload(ABC):

    def __init__(self, payload: ty.Union[int, np.ndarray, np.array]):
        self._payload = payload

    def return_payload(self) -> ty.Union[int, np.ndarray, np.array]:
        return self._payload


class AbstractPortMessage(ABC):

    def __init__(self,
                 message_header: 'AbstractPortMessageHeader',
                 message_payload: 'AbstractPortMessagePayload'):
        self._header = message_header
        self._payload = message_payload

    def payload(self) -> ty.Union[int, np.ndarray, np.array]:
        return self._payload.return_payload()

    def header(self) -> ty.Tuple['PortMessageFormat', ty.Type[int]]:
        return self._header.return_header()


class AbstractPortImplementation(ABC):
    def __init__(
        self,
        process_model: 'AbstractProcessModel',  # noqa: F821
        csp_ports: ty.List[ty.Union['AbstractCspRecvPort',
                                    'AbstractCspSendPort']] = [],
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
