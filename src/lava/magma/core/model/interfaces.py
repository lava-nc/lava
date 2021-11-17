# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC
from enum import Enum, unique
import numpy as np

from lava.magma.compiler.channels.interfaces import (
    AbstractCspRecvPort,
    AbstractCspSendPort)

from lava.magma.core.model.model import AbstractProcessModel


@unique
class PortMessageFormat(Enum):
    VECTOR_DENSE = 1
    VECTOR_SPARSE = 2
    SCALAR_DENSE = 3
    SCALAR_SPARSE = 4
    DONE = 1111
    MGMT = 2222


class AbstractPortMessage(ABC):

    def __init__(self,
                 format: 'PortMessageFormat',
                 num_elem: ty.Type[int],
                 data: ty.Union[int, np.ndarray, np.array]) -> np.ndarray:
        self._payload = np.array([format, num_elem, data], dtype=object)

    @property
    def payload(self) -> ty.Type[np.array]:
        return self._payload

    @property
    def message_type(self) -> ty.Type[np.array]:
        return self._payload[0]

    @property
    def num_elements(self) -> ty.Type[np.array]:
        return self._payload[1]

    @property
    def data(self) -> ty.Type[np.array]:
        return self._payload[2]


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
