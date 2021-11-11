# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import abstractmethod
from enum import Enum
import functools as ft

import numpy as np

from lava.magma.core.model.interfaces import (AbstractPortImplementation,
                                              AbstractPortMessage,
                                              AbstractPortMessageHeader,
                                              AbstractPortMessagePayload,
                                              PortMessageFormat)


class AbstractPyPort(AbstractPortImplementation):
    pass


class AbstractPyPortMessage(AbstractPortMessage):
    pass


class AbstractPyPortMessageHeader(AbstractPortMessageHeader):
    pass


class AbstractPyPortMessagePayload(AbstractPortMessagePayload):
    pass


class PyPortMessageHeader(AbstractPyPortMessageHeader):

    def __init__(self,
                 format: 'PortMessageFormat',
                 number_elements: ty.Type[int]):
        self._format = format
        self._number_elements = number_elements


class PyPortMessagePayload(AbstractPyPortMessagePayload):

    def __init__(self, payload: ty.Union[int, np.ndarray, np.array]):
        self._payload = payload


class PyPortMessage(AbstractPyPortMessage):

    def __init__(self,
                 message_header: 'PyPortMessageHeader',
                 message_payload: 'PyPortMessagePayload'):
        self._header = message_header
        self._payload = message_payload


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

    # Draf Impl. Receives Vector from Dense
    #
    # if not from PyOutPortVectorDense we need to
    # process the data received
    #   format, elements, payload =
    #   if format not in (PortMessageFormat.VECTOR_DENSE,
    #   PortMessageFormat.VECTOR_SPARSE, PortMessageFormat.SCALAR_DENSE,
    #   PortMessageFormat.SCALAR_SPARSE):
    #      raise AssertionError("Message format " + format + "
    #   not recognized, should be one of: PortMessageFormat")
    def recv(self) -> np.ndarray:
        messages = []
        for csp_port in self._csp_ports:
            messages.append(csp_port.recv())
        return ft.reduce(
            lambda acc, message: acc + message.payload(),
            messages,
            np.zeros(self._shape, self._d_type),
        )

    def peek(self) -> np.ndarray:
        messages = []
        for csp_port in self._csp_ports:
            messages.append(csp_port.peek())
        return ft.reduce(
            lambda acc, message: acc + message.payload(),
            messages,
            np.zeros(self._shape, self._d_type),
        )


class PyInPortVectorSparse(PyInPort):
    """Python implementation of Vector Sparse InPort
    """
    # sparse port recieves twice, for val and index.
    # if not sparse port we need to convert to dense,
    # vice-versa, possible scipy or numpy tools for conversion
    def recv(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        # Draf Impl. Receives Vector from Sparse
        #
        # if not from PyOutPortVectorSparse we need to
        # process the data received
        if self._csp_port:
            return (self._csp_port.recv(),
                    self._csp_port.recv())
        else:
            return (np.zeros(self._shape, self._d_type),
                    np.zeros(self._shape, self._d_type))

    def peek(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        if self._csp_port:
            return (self._csp_port.peek(),
                    self._csp_port.peek())
        else:
            return (np.zeros(self._shape, self._d_type),
                    np.zeros(self._shape, self._d_type))


class PyInPortScalarDense(PyInPort):
    """Python implementation of Scalar Dense InPort
    """
    def recv(self) -> int:
        # Draf Impl. Receives Scalar from Dense
        #
        # if not from PyOutPortScalarDense we need to
        # process the data received
        if self._csp_port:
            return self._csp_port.recv()
        else:
            return 0

    def peek(self) -> int:
        # Receives Scalar from Dense
        if self._csp_port:
            return self._csp_port.peek()
        else:
            return 0


class PyInPortScalarSparse(PyInPort):
    """Python implementation of Scalar Sparse InPort
    """
    def recv(self) -> ty.Tuple[int, int]:
        # Draf Impl. Receives Scalar from Sparse
        #
        # if not from PyOutPortScalarSparse we need to
        # process the data received
        if self._csp_port:
            return (self._csp_port.recv(),
                    self._csp_port.recv())
        else:
            return (0, 0)

    def peek(self) -> ty.Tuple[int, int]:
        # Receives Scalar from Sparse
        if self._csp_port:
            return (self._csp_port.peek(),
                    self._csp_port.peek())
        else:
            return (0, 0)


PyInPort.VEC_DENSE = PyInPortVectorDense
PyInPort.VEC_SPARSE = PyInPortVectorSparse
PyInPort.SCALAR_DENSE = PyInPortScalarDense
PyInPort.SCALAR_SPARSE = PyInPortScalarSparse


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
    """PyOutPort that sends VECTOR_DENSE messages"""

    def send(self, data: np.ndarray):
        """Sends VECTOR_DENSE message encoded with data
        only if port is not dangling.

        Parameters
        ----------
        data : np.ndarray
        """
        message_payload = PyPortMessagePayload(
            data
        )
        message_header = PyPortMessageHeader(
            PortMessageFormat.VECTOR_DENSE,
            data.size
        )
        message = PyPortMessage(
            message_header,
            message_payload
        )
        for csp_port in self._csp_ports:
            csp_port.send(message)


class PyOutPortVectorSparse(PyOutPort):
    """PyOutPort that sends VECTOR_SPARSE messages"""

    def send(self, data: np.ndarray, idx: np.ndarray):
        """Sends VECTOR_SPARSE message encoded with data, idx
        only if port is not dangling.

        Parameters
        ----------
        data : np.ndarray
        idx : np.ndarray
        """
        msg_data = self.interleave_message(data, idx)
        message_payload = PyPortMessagePayload(
            msg_data
        )
        message_header = PyPortMessageHeader(
            PortMessageFormat.VECTOR_SPARSE,
            msg_data.size
        )
        message = PyPortMessage(
            message_header,
            message_payload
        )
        for csp_port in self._csp_ports:
            csp_port.send(message)

    def interleave(self,
                   data: np.ndarray,
                   idx: np.ndarray) -> np.ndarray:
        """Interleave two np.ndarrays

        Parameters
        ----------
        data : np.ndarray
        idx : np.ndarray

        Returns
        -------
        np.ndarray
            interleaved np.ndarray composed of data, idx
        """
        return np.vstack((data, idx)).reshape((-1,), order='F')


class PyOutPortScalarDense(PyOutPort):
    """PyOutPort that sends SCALAR_DENSE messages"""

    def send(self, data: int):
        """Sends SCALAR_DENSE message encoded with data
        only if port is not dangling.

        Parameters
        ----------
        data : int
        """
        message_payload = PyPortMessagePayload(
            data
        )
        message_header = PyPortMessageHeader(
            PortMessageFormat.SCALAR_DENSE,
            data.size
        )
        message = PyPortMessage(
            message_header,
            message_payload
        )
        for csp_port in self._csp_ports:
            csp_port.send(message)


class PyOutPortScalarSparse(PyOutPort):
    """PyOutPort that sends SCALAR_SPARSE messages"""

    def send(self, data: int, idx: int):
        """Sends SCALAR_SPARSE message encoding data,
        idx only if port is not dangling.

        Parameters
        ----------
        data : int
        idx : int
        """
        msg_data = np.array(data, idx)
        message_payload = PyPortMessagePayload(
            msg_data
        )
        message_header = PyPortMessageHeader(
            PortMessageFormat.SCALAR_SPARSE,
            msg_data.size
        )
        message = PyPortMessage(
            message_header,
            message_payload
        )
        for csp_port in self._csp_ports:
            csp_port.send(message)


PyOutPort.VEC_DENSE = PyOutPortVectorDense
PyOutPort.VEC_SPARSE = PyOutPortVectorSparse
PyOutPort.SCALAR_DENSE = PyOutPortScalarDense
PyOutPort.SCALAR_SPARSE = PyOutPortScalarSparse


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
