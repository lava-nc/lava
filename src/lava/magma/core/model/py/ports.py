# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import abstractmethod
import functools as ft

import numpy as np
import scipy.sparse as sparse

from lava.magma.core.model.interfaces import (AbstractPortImplementation,
                                              AbstractPortMessage,
                                              PortMessageFormat)


class AbstractPyPort(AbstractPortImplementation):
    pass


class AbstractPyPortMessage(AbstractPortMessage):
    pass


class PyPortMessage(AbstractPyPortMessage):
    pass


class PyInPort(AbstractPyPort):
    """Python implementation of InPort used within AbstractPyProcessModel.
    If buffer is empty, recv() will be blocking.
    """

    VEC_DENSE: ty.Type["PyInPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyInPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyInPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyInPortScalarSparse"] = None

    def __init__(self, *args, **kwargs):
        super(PyInPort, self).__init__(*args, **kwargs)
        self.format: PortMessageFormat = None

    def recv(self):

        messages = []
        for csp_port in self._csp_ports:
            messages.append(csp_port.recv())
        self.format = messages[0][0]

        return getattr(self,
                       '_recv_' + str(self.format.name),
                       '_invalid_message_type')(messages)

    def peek(self):
        messages = []
        for csp_port in self._csp_ports:
            messages.append(csp_port.peek())
        self.format = messages[0][0]

        return getattr(self,
                       '_recv_' + str(self.format.name),
                       '_invalid_message_type')(messages)

    def probe(self) -> bool:
        pass

    @abstractmethod
    def _recv_VECTOR_DENSE(self, messages: ty.List[PyPortMessage]):
        pass

    @abstractmethod
    def _recv_VECTOR_SPARSE(self, messages: ty.List[PyPortMessage]):
        pass

    @abstractmethod
    def _recv_SCALAR_DENSE(self, messages: ty.List[PyPortMessage]):
        pass

    @abstractmethod
    def _recv_SCALAR_SPARSE(self, messages: ty.List[PyPortMessage]):
        pass

    @abstractmethod
    def _peek_VECTOR_DENSE(self, messages: ty.List[PyPortMessage]):
        pass

    @abstractmethod
    def _peek_VECTOR_SPARSE(self, messages: ty.List[PyPortMessage]):
        pass

    @abstractmethod
    def _peek_SCALAR_DENSE(self, messages: ty.List[PyPortMessage]):
        pass

    @abstractmethod
    def _peek_SCALAR_SPARSE(self, messages: ty.List[PyPortMessage]):
        pass

    def _invalid_message_type(self):
        raise ValueError(self.format)


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

    def _recv_VECTOR_DENSE(self, messages: ty.List[PyPortMessage]) -> \
            np.ndarray:
        return ft.reduce(
            lambda acc, message: acc + message.data,
            messages,
            np.zeros(messages[0].data.shape, messages[0].data.dtype)
        )

    def _recv_VECTOR_SPARSE(self, messages: ty.List[PyPortMessage]) -> \
            np.ndarray:
        reduced = ft.reduce(
            lambda acc, message: acc + message.data(),
            messages,
            np.zeros(messages[0].data.shape, messages[0].data.dtype)
        )
        uninterleaved = [reduced[idx::2] for idx in range(2)]
        return (uninterleaved[0], uninterleaved[1])
        # n_reduced = reduced.shape[1]
        # data, idx = reduced.reshape(-1, 2, n_reduced). \
        #    swapaxes(1, 2).reshape(-1, n_reduced * 2)
        # np.vstack((data, idx)).reshape((-1,), order='F')

    def _recv_SCALAR_DENSE(self, messages: ty.List[PyPortMessage]) -> \
            np.ndarray:
        raise NotImplementedError

    def _recv_SCALAR_SPARSE(self, messages: ty.List[PyPortMessage]) -> \
            np.ndarray:
        raise NotImplementedError

    def _peek_VECTOR_DENSE(self, messages: ty.List[PyPortMessage]) -> \
            np.ndarray:
        return self._recv_VECTOR_DENSE(messages)

    def _peek_VECTOR_SPARSE(self, messages: ty.List[PyPortMessage]) -> \
            np.ndarray:
        raise NotImplementedError

    def _peek_SCALAR_DENSE(self, messages: ty.List[PyPortMessage]) -> \
            np.ndarray:
        raise NotImplementedError

    def _peek_SCALAR_SPARSE(self, messages: ty.List[PyPortMessage]) -> \
            np.ndarray:
        raise NotImplementedError


class PyInPortVectorSparse(PyInPort):
    """Python implementation of Vector Sparse InPort
    """

    def _recv_VECTOR_DENSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        rows, columns, values = sparse.find(
            sparse.csr_matrix(
                ft.reduce(
                    lambda acc, message: acc + message.data(),
                    messages,
                    np.zeros(messages[0].data().shape, messages[0].data.dtype)
                )
            )
        )
        return (values, np.vstack((rows, columns)).T)

    def _recv_VECTOR_SPARSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        reduced = ft.reduce(
            lambda acc, message: acc + message.data(),
            messages,
            np.zeros(messages[0].data.shape, messages[0].data.dtype)
        )
        uninterleaved = [reduced[idx::2] for idx in range(2)]
        return (uninterleaved[0], uninterleaved[1])
        # n_reduced = reduced.shape[1]
        # data, idx = reduced.reshape(-1, 2, n_reduced). \
        #    swapaxes(1, 2).reshape(-1, n_reduced * 2)
        # np.vstack((data, idx)).reshape((-1,), order='F')

    def _recv_SCALAR_DENSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _recv_SCALAR_SPARSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _peek_VECTOR_DENSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        return self._recv_VECTOR_DENSE(messages)

    def _peek_VECTOR_SPARSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _peek_SCALAR_DENSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _peek_SCALAR_SPARSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


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
        message = PyPortMessage(
            PortMessageFormat.VECTOR_DENSE,
            data.size,
            data
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
        msg_data = self.interleave(data, idx)

        message = PyPortMessage(
            PortMessageFormat.VECTOR_SPARSE,
            msg_data.size,
            msg_data
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
        message = PyPortMessage(
            PortMessageFormat.SCALAR_DENSE,
            data.size,
            data
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
        message = PyPortMessage(
            PortMessageFormat.SCALAR_SPARSE,
            msg_data.size,
            msg_data
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
