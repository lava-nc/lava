# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import abstractmethod
from enum import Enum
import functools as ft

import numpy as np

from lava.magma.core.model.interfaces import AbstractPortImplementation


class AbstractPyPort(AbstractPortImplementation):
    pass


class PyInPort(AbstractPyPort):
    """Python implementation of InPort used within AbstractPyProcessModel.
    If buffer is empty, recv() will be blocking.
    """

    VEC_DENSE: ty.Type["PyInPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyInPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyInPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyInPortScalarSparse"] = None

    @abstractmethod
    def recv(self):
        pass

    @abstractmethod
    def peek(self):
        pass

    def probe(self) -> bool:
        pass


class PyInPortVectorDense(PyInPort):
    """Python implementation of Vector Dense InPort
    """
    def recv(self) -> np.ndarray:
        return ft.reduce(
            lambda acc, csp_port: acc + csp_port.recv(),
            self._csp_ports,
            np.zeros(self._shape, self._d_type),
        )

    def peek(self) -> np.ndarray:
        return ft.reduce(
            lambda acc, csp_port: acc + csp_port.peek(),
            self._csp_ports,
            np.zeros(self._shape, self._d_type),
        )


class PyInPortVectorSparse(PyInPort):
    """Python implementation of Vector Sparse InPort
    """
    def recv(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        if self._csp_port:
            return self._csp_port.recv()
        else:
            return (np.zeros(self._shape, self._d_type),
                    np.zeros(self._shape, self._d_type))

    def peek(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        if self._csp_port:
            return self._csp_port.peek()
        else:
            return (np.zeros(self._shape, self._d_type),
                    np.zeros(self._shape, self._d_type))


class PyInPortScalarDense(PyInPort):
    """Python implementation of Scalar Dense InPort
    """
    def recv(self) -> int:
        if self._csp_port:
            return self._csp_port.recv()
        else:
            return 0

    def peek(self) -> int:
        if self._csp_port:
            return self._csp_port.peek()
        else:
            return 0


class PyInPortScalarSparse(PyInPort):
    """Python implementation of Scalar Sparse InPort
    """
    def recv(self) -> ty.Tuple[int, int]:
        if self._csp_port:
            return self._csp_port.recv()
        else:
            return (0, 0)

    def peek(self) -> ty.Tuple[int, int]:
        if self._csp_port:
            return self._csp_port.peek()
        else:
            return (0, 0)


PyInPort.VEC_DENSE = PyInPortVectorDense
PyInPort.VEC_SPARSE = PyInPortVectorSparse
PyInPort.SCALAR_DENSE = PyInPortScalarDense
PyInPort.SCALAR_SPARSE = PyInPortScalarSparse


# ToDo: Remove... not needed anymore
class _PyInPort(Enum):
    VEC_DENSE = PyInPortVectorDense
    VEC_SPARSE = PyInPortVectorSparse
    SCALAR_DENSE = PyInPortScalarDense
    SCALAR_SPARSE = PyInPortScalarSparse


class PyOutPort(AbstractPyPort):
    """Python implementation of OutPort used within AbstractPyProcessModels."""

    VEC_DENSE: ty.Type["PyOutPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyOutPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyOutPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyOutPortScalarSparse"] = None

    @abstractmethod
    def send(self, data: ty.Union[np.ndarray, int]):
        pass

    def flush(self):
        pass


class PyOutPortVectorDense(PyOutPort):
    def send(self, data: np.ndarray):
        """Sends data only if port is not dangling."""
        for csp_port in self._csp_ports:
            csp_port.send(data)


class PyOutPortVectorSparse(PyOutPort):
    def send(self, data: np.ndarray, idx: np.ndarray):
        pass


class PyOutPortScalarDense(PyOutPort):
    def send(self, data: int):
        pass


class PyOutPortScalarSparse(PyOutPort):
    def send(self, data: int, idx: int):
        pass


PyOutPort.VEC_DENSE = PyOutPortVectorDense
PyOutPort.VEC_SPARSE = PyOutPortVectorSparse
PyOutPort.SCALAR_DENSE = PyOutPortScalarDense
PyOutPort.SCALAR_SPARSE = PyOutPortScalarSparse


# ToDo: Remove... not needed anymore
class _PyOutPort(Enum):
    VEC_DENSE = PyOutPortVectorDense
    VEC_SPARSE = PyOutPortVectorSparse
    SCALAR_DENSE = PyOutPortScalarDense
    SCALAR_SPARSE = PyOutPortScalarSparse


class PyRefPort(AbstractPyPort):
    """Python implementation of RefPort used within AbstractPyProcessModels."""

    VEC_DENSE: ty.Type["PyRefPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyRefPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyRefPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyRefPortScalarSparse"] = None

    def read(
        self,
    ) -> ty.Union[
        np.ndarray, ty.Tuple[np.ndarray, np.ndarray], int, ty.Tuple[int, int]
    ]:
        pass

    def write(
        self,
        data: ty.Union[
            np.ndarray,
            ty.Tuple[np.ndarray, np.ndarray],
            int,
            ty.Tuple[int, int],
        ],
    ):
        pass


class PyRefPortVectorDense(PyRefPort):
    def read(self) -> np.ndarray:
        pass

    def write(self, data: np.ndarray):
        pass


class PyRefPortVectorSparse(PyRefPort):
    def read(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        pass

    def write(self, data: np.ndarray, idx: np.ndarray):
        pass


class PyRefPortScalarDense(PyRefPort):
    def read(self) -> int:
        pass

    def write(self, data: int):
        pass


class PyRefPortScalarSparse(PyRefPort):
    def read(self) -> ty.Tuple[int, int]:
        pass

    def write(self, data: int, idx: int):
        pass


PyRefPort.VEC_DENSE = PyRefPortVectorDense
PyRefPort.VEC_SPARSE = PyRefPortVectorSparse
PyRefPort.SCALAR_DENSE = PyRefPortScalarDense
PyRefPort.SCALAR_SPARSE = PyRefPortScalarSparse


# ToDo: Remove... not needed anymore
class _PyRefPort(Enum):
    VEC_DENSE = PyRefPortVectorDense
    VEC_SPARSE = PyRefPortVectorSparse
    SCALAR_DENSE = PyRefPortScalarDense
    SCALAR_SPARSE = PyRefPortScalarSparse
