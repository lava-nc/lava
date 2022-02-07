# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import abstractmethod
import functools as ft
import numpy as np

from lava.magma.compiler.channels.interfaces import AbstractCspPort
from lava.magma.compiler.channels.pypychannel import CspSendPort, CspRecvPort
from lava.magma.core.model.interfaces import AbstractPortImplementation
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.runtime.mgmt_token_enums import enum_to_np, enum_equal


class AbstractPyPort(AbstractPortImplementation):
    """Abstract class for ports implemented in python. A port must have one or
    multiple csp ports."""
    @property
    @abstractmethod
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """
        Abstract property to get all csp ports of a port.

        Returns
        -------
        A list of all csp ports used by the port.
        """
        pass


class AbstractPyIOPort(AbstractPyPort):
    """Abstract class of an input/output port implemented in python.

    Parameters
    ----------
    csp_ports : list
        A list of csp ports used by this IO port.

    process_model : AbstractProcessModel
        The process model used by the process of the port.

    shape : tuple, default=tuple()
        The shape of the port.

    d_type: type, default=int
        The data type of the port.

    Attributes
    ----------
    _csp_ports : list
        A list of csp ports used by this IO port.
    """
    def __init__(self,
                 csp_ports: ty.List[AbstractCspPort],
                 process_model: AbstractProcessModel,
                 shape: ty.Tuple[int, ...] = tuple(),
                 d_type: type = int):

        self._csp_ports = csp_ports
        super().__init__(process_model, shape, d_type)

    @property
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Property to get all csp ports used by the port.

        Returns
        -------
        A list of all csp ports used by the port.
        """
        return self._csp_ports


class PyInPort(AbstractPyIOPort):
    """Python implementation of InPort used within AbstractPyProcessModel.
    If buffer is empty, recv() will be blocking.

    Attributes
    ----------
    VEC_DENSE : PyInPortVectorDense, default=None
        Specifies that dense data vectors should be sent on this port.

    VEC_SPARSE : PyInPortVectorSparse, default=None
        Specifies that sparse data vectors should be sent on this port.

    SCALAR_DENSE : PyInPortScalarDense, default=None
        Specifies that dense scalars should be sent on this port.

    SCALAR_SPARSE : PyInPortScalarSparse, default=None
        Specifies that sparse scalars should be sent on this port.
    """

    VEC_DENSE: ty.Type["PyInPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyInPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyInPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyInPortScalarSparse"] = None

    @abstractmethod
    def recv(self):
        """Abstract method to receive data (vectors/scalars) sent from connected
        out ports (source ports). Removes the data from the channel.

        Returns
        -------
        The point-wise added vectors or scalars received from connected ports.
        """
        pass

    @abstractmethod
    def peek(self):
        """Abstract method to receive data (vectors/scalars) sent from connected
        out ports (source ports). Keeps the data on the channel.

        Returns
        -------
        The point-wise added vectors or scalars received from connected ports.
        """
        pass

    def probe(self) -> bool:
        """Method to check (probe) if there is data (vectors/scalars)
        to receive from connected out ports (source ports).

        Returns
        -------
        result : bool
             Returns True only when probe returns True for all csp_ports.

        """
        return ft.reduce(
            lambda acc, csp_port: acc and csp_port.probe(),
            self.csp_ports,
            True,
        )


class PyInPortVectorDense(PyInPort):
    def recv(self) -> np.ndarray:
        """Method to receive data (vectors/scalars) sent from connected
        out ports (source ports). Removes the data from the channel.

        Returns
        -------
        result : ndarray of shape _shape
            The point-wise added vectors received from connected ports.
        """
        return ft.reduce(
            lambda acc, csp_port: acc + csp_port.recv(),
            self.csp_ports,
            np.zeros(self._shape, self._d_type),
        )

    def peek(self) -> np.ndarray:
        """Method to receive data (vectors) sent from connected
        out ports (source ports). Keeps the data on the channel.

        Returns
        -------
        result : ndarray of shape _shape
        The point-wise added vectors received from connected ports.
        """
        return ft.reduce(
            lambda acc, csp_port: acc + csp_port.peek(),
            self.csp_ports,
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


class PyOutPort(AbstractPyIOPort):
    """Python implementation of OutPort used within AbstractPyProcessModels.

    Attributes
    ----------
    VEC_DENSE : PyOutPortVectorDense, default=None
        Specifies that dense data vectors should be sent on this port.

    VEC_SPARSE : PyOutPortVectorSparse, default=None
        Specifies that sparse data vectors should be sent on this port.

    SCALAR_DENSE : PyOutPortScalarDense, default=None
        Specifies that dense scalars should be sent on this port.

    SCALAR_SPARSE : PyOutPortScalarSparse, default=None
        Specifies that sparse scalars should be sent on this port.
    """

    VEC_DENSE: ty.Type["PyOutPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyOutPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyOutPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyOutPortScalarSparse"] = None

    @abstractmethod
    def send(self, data: ty.Union[np.ndarray, int]):
        """Abstract method to send data to the connected in port (target).

        Parameters
        ----------
        data : ndarray or int
            The data (vector or scalar) to be sent to the in port (target).
        """
        pass

    def flush(self):
        """TBD"""
        pass


class PyOutPortVectorDense(PyOutPort):
    """Python implementation of PyOutPort for dense vector data."""

    def send(self, data: np.ndarray):
        """Abstract method to send data to the connected in port (target).

        Sends data only if port is not dangling.

        Parameters
        ----------
        data : ndarray
            The data vector to be sent to the in port (target).
        """
        for csp_port in self.csp_ports:
            csp_port.send(data)


class PyOutPortVectorSparse(PyOutPort):
    """Python implementation of PyOutPort for sparse vector data."""
    def send(self, data: np.ndarray, idx: np.ndarray):
        """TBD"""
        pass


class PyOutPortScalarDense(PyOutPort):
    """Python implementation of PyOutPort for dense scalar data."""
    def send(self, data: int):
        """TBD"""
        pass


class PyOutPortScalarSparse(PyOutPort):
    """Python implementation of PyOutPort for sparse scalar data."""
    def send(self, data: int, idx: int):
        """TBD"""
        pass


PyOutPort.VEC_DENSE = PyOutPortVectorDense
PyOutPort.VEC_SPARSE = PyOutPortVectorSparse
PyOutPort.SCALAR_DENSE = PyOutPortScalarDense
PyOutPort.SCALAR_SPARSE = PyOutPortScalarSparse


class VarPortCmd:
    """Helper class to specify constants."""
    GET = enum_to_np(0)
    SET = enum_to_np(1)


class PyRefPort(AbstractPyPort):
    """Python implementation of RefPort used within AbstractPyProcessModels.

    Parameters
    ----------
    csp_send_port : CspSendPort or None
        Csp port used to send data to the referenced in port (target).

    csp_recv_port: CspRecvPort or None
        Csp port used to receive data from the referenced port (source).

    process_model : AbstractProcessModel
        The process model used by the process of the port.

    shape : tuple, default=tuple()
        The shape of the port.

    d_type: type, default=int
        The data type of the port.

    Attributes
    ----------
    VEC_DENSE : PyRefPortVectorDense, default=None
        Specifies that dense data vectors should be sent on this port.

    VEC_SPARSE : PyRefPortVectorSparse, default=None
        Specifies that sparse data vectors should be sent on this port.

    SCALAR_DENSE : PyRefPortScalarDense, default=None
        Specifies that dense scalars should be sent on this port.

    SCALAR_SPARSE : PyRefPortScalarSparse, default=None
        Specifies that sparse scalars should be sent on this port.

    _csp_send_port : CspSendPort
        Csp port used to send data to the referenced in port (target).

    _csp_recv_port : CspRecvPort
        Csp port used to receive data from the referenced port (source).
    """

    VEC_DENSE: ty.Type["PyRefPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyRefPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyRefPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyRefPortScalarSparse"] = None

    def __init__(self,
                 csp_send_port: ty.Optional[CspSendPort],
                 csp_recv_port: ty.Optional[CspRecvPort],
                 process_model: AbstractProcessModel,
                 shape: ty.Tuple[int, ...] = tuple(),
                 d_type: type = int):
        self._csp_recv_port = csp_recv_port
        self._csp_send_port = csp_send_port
        super().__init__(process_model, shape, d_type)

    @property
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Property to get all csp ports used by the port.

        Returns
        -------
        A list of all csp ports used by the port.
        """
        if self._csp_send_port is not None and self._csp_recv_port is not None:
            return [self._csp_send_port, self._csp_recv_port]
        else:
            # In this case the port was not connected
            return []

    @abstractmethod
    def read(
            self,
    ) -> ty.Union[
        np.ndarray, ty.Tuple[np.ndarray, np.ndarray], int, ty.Tuple[int, int]
    ]:
        """Abstract method to request and return data from a VarPort.
        Returns
        -------
        The value of the referenced var.
        """
        pass

    @abstractmethod
    def write(
            self,
            data: ty.Union[
                np.ndarray,
                ty.Tuple[np.ndarray, np.ndarray],
                int,
                ty.Tuple[int, int],
            ],
    ):
        """Abstract method to write data to a VarPort to set its Var.

        Parameters
        ----------
        data : ndarray, tuple of ndarray, int, tuple of int
            The new value of the referenced Var.
        """
        pass


class PyRefPortVectorDense(PyRefPort):
    """Python implementation of RefPort for dense vector data."""
    def read(self) -> np.ndarray:
        """Method to request and return data from a referenced Var.

        Returns
        -------
        result : ndarray of shape _shape
            The value of the referenced var.
        """
        if self._csp_send_port and self._csp_recv_port:
            header = np.ones(self._csp_send_port.shape) * VarPortCmd.GET
            self._csp_send_port.send(header)

            return self._csp_recv_port.recv()

        return np.zeros(self._shape, self._d_type)

    def write(self, data: np.ndarray):
        """Abstract method to write data to a VarPort to set its Var.

        Parameters
        ----------
        data : ndarray
            The data to send via _csp_send_port.
        """
        if self._csp_send_port:
            header = np.ones(self._csp_send_port.shape) * VarPortCmd.SET
            self._csp_send_port.send(header)
            self._csp_send_port.send(data)


class PyRefPortVectorSparse(PyRefPort):
    """Python implementation of RefPort for sparse vector data."""
    def read(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """TBD"""
        pass

    def write(self, data: np.ndarray, idx: np.ndarray):
        """TBD"""
        pass


class PyRefPortScalarDense(PyRefPort):
    """Python implementation of RefPort for dense scalar data."""
    def read(self) -> int:
        """TBD"""
        pass

    def write(self, data: int):
        """TBD"""
        pass


class PyRefPortScalarSparse(PyRefPort):
    """Python implementation of RefPort for sparse scalar data."""
    def read(self) -> ty.Tuple[int, int]:
        """TBD"""
        pass

    def write(self, data: int, idx: int):
        """TBD"""
        pass


PyRefPort.VEC_DENSE = PyRefPortVectorDense
PyRefPort.VEC_SPARSE = PyRefPortVectorSparse
PyRefPort.SCALAR_DENSE = PyRefPortScalarDense
PyRefPort.SCALAR_SPARSE = PyRefPortScalarSparse


class PyVarPort(AbstractPyPort):
    """Python implementation of VarPort used within AbstractPyProcessModel.

    Parameters
    ----------
    var_name : str
        The name of the Var related to this VarPort.

    csp_send_port : CspSendPort or None
        Csp port used to send data to the referenced in port (target).

    csp_recv_port: CspRecvPort or None
        Csp port used to receive data from the referenced port (source).

    process_model : AbstractProcessModel
        The process model used by the process of the port.

    shape : tuple, default=tuple()
        The shape of the port.

    d_type: type, default=int
        The data type of the port.

    Attributes
    ----------
    VEC_DENSE : PyRefPortVectorDense, default=None
        Specifies that dense data vectors should be sent on this port.

    VEC_SPARSE : PyRefPortVectorSparse, default=None
        Specifies that sparse data vectors should be sent on this port.

    SCALAR_DENSE : PyRefPortScalarDense, default=None
        Specifies that dense scalars should be sent on this port.

    SCALAR_SPARSE : PyRefPortScalarSparse, default=None
        Specifies that sparse scalars should be sent on this port.

    var_name : str
        The name of the Var related to this VarPort.

    _csp_send_port : CspSendPort
        Csp port used to send data to the referenced in port (target).

    _csp_recv_port : CspRecvPort
        Csp port used to receive data from the referenced port (source).
    """

    VEC_DENSE: ty.Type["PyVarPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyVarPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyVarPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyVarPortScalarSparse"] = None

    def __init__(self,
                 var_name: str,
                 csp_send_port: ty.Optional[CspSendPort],
                 csp_recv_port: ty.Optional[CspRecvPort],
                 process_model: AbstractProcessModel,
                 shape: ty.Tuple[int, ...] = tuple(),
                 d_type: type = int):
        self._csp_recv_port = csp_recv_port
        self._csp_send_port = csp_send_port
        self.var_name = var_name
        super().__init__(process_model, shape, d_type)

    @property
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Property to get all csp ports used by the port.

        Returns
        -------
        A list of all csp ports used by the port.
        """
        if self._csp_send_port is not None and self._csp_recv_port is not None:
            return [self._csp_send_port, self._csp_recv_port]
        else:
            # In this case the port was not connected
            return []

    @abstractmethod
    def service(self):
        """Abstract method to set the received value to the given Var or sends
        the value of the Var to the _csp_send_port, depending on the received
        header information of the _csp_recv_port."""
        pass


class PyVarPortVectorDense(PyVarPort):
    """Python implementation of VarPort for dense vector data."""
    def service(self):
        """Abstract method to set the received value to the given Var or sends
        the value of the Var to the _csp_send_port, depending on the received
        header information of the _csp_recv_port."""

        # Inspect incoming data
        if self._csp_send_port is not None and self._csp_recv_port is not None:
            if self._csp_recv_port.probe():
                # If received data is a matrix, flatten and take the first
                # element as cmd
                cmd = enum_to_np((self._csp_recv_port.recv()).flatten()[0])

                # Set the value of the Var with the given data
                if enum_equal(cmd, VarPortCmd.SET):
                    data = self._csp_recv_port.recv()
                    setattr(self._process_model, self.var_name, data)
                elif enum_equal(cmd, VarPortCmd.GET):
                    data = getattr(self._process_model, self.var_name)
                    self._csp_send_port.send(data)
                else:
                    raise ValueError(f"Wrong Command Info Received : {cmd}")


class PyVarPortVectorSparse(PyVarPort):
    """Python implementation of VarPort for sparse vector data."""
    def recv(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """TBD"""
        pass

    def peek(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """TBD"""
        pass


class PyVarPortScalarDense(PyVarPort):
    """Python implementation of VarPort for dense scalar data."""
    def recv(self) -> int:
        """TBD"""
        pass

    def peek(self) -> int:
        """TBD"""
        pass


class PyVarPortScalarSparse(PyVarPort):
    """Python implementation of VarPort for sparse scalar data."""
    def recv(self) -> ty.Tuple[int, int]:
        """TBD"""
        pass

    def peek(self) -> ty.Tuple[int, int]:
        """TBD"""
        pass


PyVarPort.VEC_DENSE = PyVarPortVectorDense
PyVarPort.VEC_SPARSE = PyVarPortVectorSparse
PyVarPort.SCALAR_DENSE = PyVarPortScalarDense
PyVarPort.SCALAR_SPARSE = PyVarPortScalarSparse


class RefVarTypeMapping:
    """Class to get the mapping of PyRefPort types to PyVarPort types.

    Attributes
    ----------
    mapping : dict
        Dictionary containing the mapping of PyRefPort types to PyVarPort types.

    """

    mapping: ty.Dict[PyRefPort, PyVarPort] = {
        PyRefPortVectorDense: PyVarPortVectorDense,
        PyRefPortVectorSparse: PyVarPortVectorSparse,
        PyRefPortScalarDense: PyVarPortScalarDense,
        PyRefPortScalarSparse: PyVarPortScalarSparse}

    @classmethod
    def get(cls, ref_port: PyRefPort):
        """Class method to return the PyVarPort type given the PyRefPort type.

        Parameters
        ----------
        ref_port : PyRefPort
            PyRefPort type to be mapped to a PyVarPort type.

        Returns
        -------
        result : PyVarPort
            PyVarPort type corresponding to PyRefPort type given by ref_port.

        """
        return cls.mapping[ref_port]
