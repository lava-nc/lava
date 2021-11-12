# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import abstractmethod
import functools as ft
import numpy as np

from lava.magma.compiler.channels.pypychannel import CspSendPort, CspRecvPort
from lava.magma.core.model.interfaces import AbstractPortImplementation
from lava.magma.runtime.mgmt_token_enums import enum_to_np


class AbstractPyPort(AbstractPortImplementation):
    pass


class PyInPort(AbstractPyPort):
    """Python implementation of InPort used within AbstractPyProcessModel.
    If buffer is empty, recv() will be blocking.
    """

    def __init__(self, csp_recv_ports: ty.List[CspRecvPort], *args):
        self._csp_recv_ports = csp_recv_ports
        super().__init__(*args)

    VEC_DENSE: ty.Type["PyInPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyInPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyInPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyInPortScalarSparse"] = None

    def start(self):
        for csp_port in self._csp_recv_ports:
            csp_port.start()

    def join(self):
        for csp_port in self._csp_recv_ports:
            csp_port.join()

    @abstractmethod
    def recv(self):
        pass

    @abstractmethod
    def peek(self):
        pass

    def probe(self) -> bool:
        pass


class PyInPortVectorDense(PyInPort):
    def recv(self) -> np.ndarray:
        return ft.reduce(
            lambda acc, csp_port: acc + csp_port.recv(),
            self._csp_recv_ports,
            np.zeros(self._shape, self._d_type),
        )

    def peek(self) -> np.ndarray:
        return ft.reduce(
            lambda acc, csp_port: acc + csp_port.peek(),
            self._csp_recv_ports,
            np.zeros(self._shape, self._d_type),
        )


class PyInPortVectorSparse(PyInPort):
    def recv(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        pass

    def peek(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        pass


class PyInPortScalarDense(PyInPort):
    def recv(self) -> int:
        pass

    def peek(self) -> int:
        pass


class PyInPortScalarSparse(PyInPort):
    def recv(self) -> ty.Tuple[int, int]:
        pass

    def peek(self) -> ty.Tuple[int, int]:
        pass


PyInPort.VEC_DENSE = PyInPortVectorDense
PyInPort.VEC_SPARSE = PyInPortVectorSparse
PyInPort.SCALAR_DENSE = PyInPortScalarDense
PyInPort.SCALAR_SPARSE = PyInPortScalarSparse


class PyOutPort(AbstractPyPort):
    """Python implementation of OutPort used within AbstractPyProcessModels."""

    def __init__(self, csp_send_ports: ty.List[CspSendPort], *args):
        self._csp_send_ports = csp_send_ports
        super().__init__(*args)

    VEC_DENSE: ty.Type["PyOutPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyOutPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyOutPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyOutPortScalarSparse"] = None

    def start(self):
        for csp_port in self._csp_send_ports:
            csp_port.start()

    def join(self):
        for csp_port in self._csp_send_ports:
            csp_port.join()

    @abstractmethod
    def send(self, data: ty.Union[np.ndarray, int]):
        pass

    def flush(self):
        pass


class PyOutPortVectorDense(PyOutPort):
    def send(self, data: np.ndarray):
        """Sends data only if port is not dangling."""
        for csp_port in self._csp_send_ports:
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


class VarPortCmd:
    GET = enum_to_np(0)
    SET = enum_to_np(1)


class PyRefPort(AbstractPyPort):
    """Python implementation of RefPort used within AbstractPyProcessModels."""

    def __init__(self,
                 csp_send_port: ty.Optional[CspSendPort],
                 csp_recv_port: ty.Optional[CspRecvPort], *args):
        self._csp_recv_port = csp_recv_port
        self._csp_send_port = csp_send_port
        super().__init__(*args)

    VEC_DENSE: ty.Type["PyRefPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyRefPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyRefPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyRefPortScalarSparse"] = None

    def start(self):
        if self._csp_send_port is not None:
            self._csp_send_port.start()
        if self._csp_recv_port is not None:
            self._csp_recv_port.start()

    def join(self):
        if self._csp_send_port is not None:
            self._csp_send_port.join()
        if self._csp_recv_port is not None:
            self._csp_recv_port.join()

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
    """Python implementation of InPort used within AbstractPyProcessModel.
    If buffer is empty, recv() will be blocking.
    """

    def __init__(self,
                 var_name: str,
                 csp_send_port: ty.Optional[CspSendPort],
                 csp_recv_port: ty.Optional[CspRecvPort], *args):
        self._csp_recv_port = csp_recv_port
        self._csp_send_port = csp_send_port
        self.var_name = var_name
        super().__init__(*args)

    VEC_DENSE: ty.Type["PyVarPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyVarPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyVarPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyVarPortScalarSparse"] = None

    def start(self):
        if self._csp_send_port is not None:
            self._csp_send_port.start()
        if self._csp_recv_port is not None:
            self._csp_recv_port.start()

    def join(self):
        if self._csp_send_port is not None:
            self._csp_send_port.join()
        if self._csp_recv_port is not None:
            self._csp_recv_port.join()

    def service(self):
        pass


class PyVarPortVectorDense(PyVarPort):
    def service(self):
        """Sets the received value to the given var or sends the value of the
        var to the csp_send_port, depending on the received header information
        of the csp_recv_port."""

        # Inspect incoming data
        if self._csp_send_port is not None and self._csp_recv_port is not None:
            while self._csp_recv_port.probe():
                cmd = enum_to_np(self._csp_recv_port.recv()[0])

                # Set the value of the Var with the given data
                if np.array_equal(cmd, VarPortCmd.SET):
                    data = self._csp_recv_port.recv()
                    setattr(self._process_model, self.var_name, data)
                elif np.array_equal(cmd, VarPortCmd.GET):
                    data = getattr(self._process_model, self.var_name)
                    self._csp_send_port.send(data)
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
