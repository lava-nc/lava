# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import abstractmethod
import functools as ft
import numpy as np
import scipy.sparse as sparse

from lava.magma.compiler.channels.interfaces import AbstractCspPort
from lava.magma.compiler.channels.pypychannel import CspSendPort, CspRecvPort
from lava.magma.core.model.interfaces import (AbstractPortImplementation,
                                              AbstractPortMessage,
                                              PortMessageFormat)
from lava.magma.runtime.mgmt_token_enums import enum_to_message, enum_equal


class AbstractPyPort(AbstractPortImplementation):

    @property
    @abstractmethod
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Returns all csp ports of the port."""
        pass


class PyPortMessage(AbstractPortMessage):
    pass


class PyInPort(AbstractPyPort):
    """Python implementation of InPort used within AbstractPyProcessModel.
    If buffer is empty, recv() will be blocking.
    """

    VEC_DENSE: ty.Type["PyInPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyInPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyInPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyInPortScalarSparse"] = None

    def __init__(self, csp_recv_ports: ty.List[CspRecvPort], *args, **kwargs):
        self._csp_recv_ports = (
            csp_recv_ports if isinstance(csp_recv_ports, list) else [
                csp_recv_ports]
        )
        super(PyInPort, self).__init__(*args, **kwargs)

    @property
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Returns all csp ports of the port."""
        return self._csp_recv_ports

    def recv(self):
        """Receives data from connected ports and translates
        it according to its port type and returns it
        """
        messages = []
        format = ""
        for csp_port in self.csp_ports:
            msg = csp_port.recv()
            message = PyPortMessage(msg[0], msg[1], msg[2])
            messages.append(message)
        if len(messages) > 0:
            format = messages[0].message_type.name

        # Select method based on the format of the first message
        # in the form of _recv_<self.format.name>
        return getattr(self,
                       '_recv_' + str(format),
                       '_invalid_message_type')(messages)

    def peek(self):
        """Receives data from connected ports (does not remove
        it from the queue and translates it according to
        its port type and returns it
        """
        messages = []
        format = ""
        for csp_port in self.csp_ports:
            msg = csp_port.peek()
            message = PyPortMessage(msg[0], msg[1], msg[2])
            messages.append(message)
        if len(messages) > 0:
            format = messages[0].message_type.name

        # Select method based on the format of the first message
        # in the form of _recv_<self.format.name>
        return getattr(self,
                       '_recv_' + str(format),
                       '_invalid_message_type')(messages)

    def probe(self) -> bool:
        """Executes probe method of all csp ports and accumulates the returned
        bool values with AND operation. The accumulator acc is initialized to
        True.

        Returns
        -------
        bool
            Returns True only when probe returns True for all _csp_recv_ports.
        """
        return ft.reduce(
            lambda acc, csp_port: acc and csp_port.probe(),
            self.csp_ports,
            True,
        )

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
        # uninterleaved = [reduced[idx::2] for idx in range(2)]
        return (reduced[0], reduced[1])
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
        data = sparse.csr_matrix(
            ft.reduce(
                lambda acc, message: acc + message.data,
                messages,
                np.zeros(messages[0].data.shape, messages[0].data.dtype)
            )
        )
        return (data.data, data.indices)

    def _recv_VECTOR_SPARSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        reduced = ft.reduce(
            lambda acc, message: acc + message.data,
            messages,
            np.zeros(messages[0].data.shape, messages[0].data.dtype)
        )
        # uninterleaved = [reduced[idx::2] for idx in range(2)]
        return ([reduced[1],
                 reduced[2]],
                [reduced[0]]
                )

    def _recv_SCALAR_DENSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        data = sparse.csr_matrix(
            ft.reduce(
                lambda acc, message: acc + message.data,
                messages,
                np.zeros(messages[0].data.shape, messages[0].data.dtype)
            )
        )
        return (data.data, data.indices)

    def _recv_SCALAR_SPARSE(self, messages: ty.List[PyPortMessage]) -> \
            ty.Tuple[np.ndarray, np.ndarray]:
        reduced = ft.reduce(
            lambda acc, message: acc + message.data,
            messages,
            np.zeros(messages[0].data.shape, messages[0].data.dtype)
        )
        # uninterleaved = [reduced[idx::2] for idx in range(2)]
        return ([reduced[1],
                 reduced[2]],
                [reduced[0]]
                )

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

    def __init__(self, csp_send_ports: ty.List[CspSendPort], *args, **kwargs):
        self._csp_send_ports = (
            csp_send_ports if isinstance(csp_send_ports, list) else [
                csp_send_ports]
        )
        super(PyOutPort, self).__init__(*args, **kwargs)

    @property
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Returns all csp ports of the port."""
        return self._csp_send_ports

    @abstractmethod
    def send(self, data: ty.Union[np.ndarray, int]):
        pass

    def flush(self):
        pass


class PyOutPortVectorDense(PyOutPort):
    """PyOutPort that sends VECTOR_DENSE messages"""

    def send(self, data: ty.Union[np.ndarray, int]):
        """Sends VECTOR_DENSE message encoded with data
        only if port is not dangling.
        Parameters
        ----------
        data : [np.ndarray, int]
        """
        message = PyPortMessage(
            PortMessageFormat.VECTOR_DENSE,
            data.size,
            data
        )
        for csp_port in self.csp_ports:
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
        # msg_data = [data, idx]

        message = PyPortMessage(
            PortMessageFormat.VECTOR_SPARSE,
            msg_data.size,
            msg_data
        )
        for csp_port in self.csp_ports:
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
        # return np.dstack((data, idx)).reshape(data.shape[0], -1)


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
        for csp_port in self.csp_ports:
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
        msg_data = np.array((data, idx))
        message = PyPortMessage(
            PortMessageFormat.SCALAR_SPARSE,
            msg_data.size,
            msg_data
        )
        for csp_port in self.csp_ports:
            csp_port.send(message)


PyOutPort.VEC_DENSE = PyOutPortVectorDense
PyOutPort.VEC_SPARSE = PyOutPortVectorSparse
PyOutPort.SCALAR_DENSE = PyOutPortScalarDense
PyOutPort.SCALAR_SPARSE = PyOutPortScalarSparse


class VarPortCmd:
    GET = enum_to_message(0)
    SET = enum_to_message(1)


class PyRefPort(AbstractPyPort):
    """Python implementation of RefPort used within AbstractPyProcessModels."""

    VEC_DENSE: ty.Type["PyRefPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyRefPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyRefPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyRefPortScalarSparse"] = None

    def __init__(self,
                 csp_send_port: ty.Optional[CspSendPort],
                 csp_recv_port: ty.Optional[CspRecvPort], *args):
        self._csp_recv_port = csp_recv_port
        self._csp_send_port = csp_send_port
        super().__init__(*args)

    @property
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Returns all csp ports of the port."""
        if self._csp_send_port is not None and self._csp_recv_port is not None:
            return [self._csp_send_port, self._csp_recv_port]
        else:
            # In this case the port was not connected
            return []

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
        """Requests the data from a VarPort and returns the data."""
        if self._csp_send_port and self._csp_recv_port:
            header = np.ones(self._csp_send_port.shape) * VarPortCmd.GET
            self._csp_send_port.send(header)

            return self._csp_recv_port.recv()

        return np.zeros(self._shape, self._d_type)

    def write(self, data: np.ndarray):
        """Sends the data to a VarPort to set its Var."""
        if self._csp_send_port:
            header = np.ones(self._csp_send_port.shape) * VarPortCmd.SET
            self._csp_send_port.send(header)
            self._csp_send_port.send(data)


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


class PyVarPort(AbstractPyPort):
    """Python implementation of VarPort used within AbstractPyProcessModel.
    """

    VEC_DENSE: ty.Type["PyVarPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyVarPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyVarPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyVarPortScalarSparse"] = None

    def __init__(self,
                 var_name: str,
                 csp_send_port: ty.Optional[CspSendPort],
                 csp_recv_port: ty.Optional[CspRecvPort], *args):
        self._csp_recv_port = csp_recv_port
        self._csp_send_port = csp_send_port
        self.var_name = var_name
        super().__init__(*args)

    @property
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Returns all csp ports of the port."""
        if self._csp_send_port is not None and self._csp_recv_port is not None:
            return [self._csp_send_port, self._csp_recv_port]
        else:
            # In this case the port was not connected
            return []

    def service(self):
        pass


class PyVarPortVectorDense(PyVarPort):
    def service(self):
        """Sets the received value to the given var or sends the value of the
        var to the csp_send_port, depending on the received header information
        of the csp_recv_port."""

        # Inspect incoming data
        if self._csp_send_port is not None and self._csp_recv_port is not None:
            if self._csp_recv_port.probe():
                # If received data is a matrix, flatten and take the first
                # element as cmd
                cmd = enum_to_message(
                    (self._csp_recv_port.recv().data).flatten()[0]
                )

                # Set the value of the Var with the given data
                if enum_equal(cmd, VarPortCmd.SET):
                    data = self._csp_recv_port.recv().data
                    setattr(self._process_model, self.var_name, data)
                elif enum_equal(cmd, VarPortCmd.GET):
                    data = getattr(self._process_model, self.var_name)
                    self._csp_send_port.send(enum_to_message(data))
                else:
                    raise ValueError(f"Wrong Command Info Received : {cmd}")


class PyVarPortVectorSparse(PyVarPort):
    def recv(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        pass

    def peek(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        pass


class PyVarPortScalarDense(PyVarPort):
    def recv(self) -> int:
        pass

    def peek(self) -> int:
        pass


class PyVarPortScalarSparse(PyVarPort):
    def recv(self) -> ty.Tuple[int, int]:
        pass

    def peek(self) -> ty.Tuple[int, int]:
        pass


PyVarPort.VEC_DENSE = PyVarPortVectorDense
PyVarPort.VEC_SPARSE = PyVarPortVectorSparse
PyVarPort.SCALAR_DENSE = PyVarPortScalarDense
PyVarPort.SCALAR_SPARSE = PyVarPortScalarSparse


class RefVarTypeMapping:
    """Class to get the mapping of PyRefPort types to PyVarPortTypes."""

    mapping: ty.Dict[PyRefPort, PyVarPort] = {
        PyRefPortVectorDense: PyVarPortVectorDense,
        PyRefPortVectorSparse: PyVarPortVectorSparse,
        PyRefPortScalarDense: PyVarPortScalarDense,
        PyRefPortScalarSparse: PyVarPortScalarSparse}

    @classmethod
    def get(cls, ref_port: PyRefPort):
        return cls.mapping[ref_port]
