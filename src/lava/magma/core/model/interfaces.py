# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
<<<<<<< HEAD
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
=======
from abc import ABC, abstractmethod
from lava.magma.compiler.channels.interfaces import AbstractCspPort
>>>>>>> main


class AbstractPortImplementation(ABC):
    def __init__(
        self,
<<<<<<< HEAD
        process_model: 'AbstractProcessModel',  # noqa: F821
        csp_ports: ty.List[ty.Union['AbstractCspRecvPort',
                                    'AbstractCspSendPort']] = [],
=======
        process_model: "AbstractProcessModel",  # noqa: F821
>>>>>>> main
        shape: ty.Tuple[int, ...] = tuple(),
        d_type: type = int,
    ):
        self._process_model = process_model
        self._shape = shape
        self._d_type = d_type

    @property
    @abstractmethod
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Returns all csp ports of the port."""
        pass

    def start(self):
        """Start all csp ports."""
        for csp_port in self.csp_ports:
            csp_port.start()

    def join(self):
        """Join all csp ports"""
        for csp_port in self.csp_ports:
            csp_port.join()
