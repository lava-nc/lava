# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC
from multiprocessing import shared_memory

from enum import Enum, unique
import numpy as np

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
                 format: ty.Union[int, 'PortMessageFormat'],
                 num_elem: ty.Type[int],
                 data: ty.Union[int, np.ndarray, np.array],
                 shm_buf_name: str = "") -> np.ndarray:
        if isinstance(format, PortMessageFormat):
            format_value = format.value
        elif isinstance(format, int):
            self._format = PortMessageFormat(format)
            format_value = format
        else:
            raise AssertionError("PortMessageFormat is invalid")
        if shm_buf_name != "":
            shm = shared_memory.SharedMemory(name=shm_buf_name)

            self._payload = np.array(
                [format_value,
                 num_elem,
                 data],
                copy=True,
                buffer=shm.buf
            )
        else:
            self._payload = np.array(
                [format_value,
                 num_elem,
                 data]
            )

    @property
    def payload(self) -> ty.Type[np.array]:
        return self._payload

    @property
    def message_type(self) -> ty.Type[np.array]:
        return self._format

    @property
    def num_elements(self) -> ty.Type[np.array]:
        return self._payload[1]

    @property
    def data(self) -> ty.Union[int, np.ndarray, np.array]:
        return self._payload[2]


class AbstractPortImplementation(ABC):
    def __init__(
        self,
        process_model: "AbstractProcessModel" = None,  # noqa: F821
        shape: ty.Tuple[int, ...] = tuple(),
        d_type: type = int,
        shared_mem_name: str = ""
    ):
        self._process_model = process_model
        self._shared_mem_name = shared_mem_name

        self._shape = shape
        self._d_type = d_type

    def start(self):
        """Start all csp ports."""
        for csp_port in self.csp_ports:
            csp_port.start()

    def join(self):
        """Join all csp ports"""
        for csp_port in self.csp_ports:
            csp_port.join()
