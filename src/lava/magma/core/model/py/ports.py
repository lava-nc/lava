# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import functools as ft
import typing as ty
from abc import ABC, abstractmethod

import numpy as np

from lava.magma.compiler.channels.interfaces import AbstractCspPort
from lava.magma.compiler.channels.pypychannel import CspSendPort, CspRecvPort
from lava.magma.core.model.interfaces import AbstractPortImplementation
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.runtime.mgmt_token_enums import enum_to_np, enum_equal


class AbstractPyPort(AbstractPortImplementation):
    """Abstract class for Ports implemented in Python.

    Ports at the Process level provide an interface to connect
    Processes with each other. Once two Processes have been connected by Ports,
    they can exchange data.
    Lava provides four types of Ports: InPorts, OutPorts, RefPorts and VarPorts.
    An OutPort of a Process can be connected to one or multiple InPorts of other
    Processes to transfer data from the OutPort to the InPorts. A RefPort of a
    Process can be connected to a VarPort of another Process. The difference to
    In-/OutPorts is that a VarPort is directly linked to a Var and via a
    RefPort the Var can be directly modified from a different Process.
    To exchange data, PyPorts provide an interface to send and receive messages
    via channels implemented by a backend messaging infrastructure, which has
    been inspired by the Communicating Sequential Processes (CSP) paradigm.
    Thus, a channel denotes a CSP channel of the messaging infrastructure and
    CSP Ports denote the low level ports also used in the messaging
    infrastructure. PyPorts are the implementation for message exchange in
    Python, using the low level CSP Ports of the backend messaging
    infrastructure. A PyPort may have one or multiple connection to other
    PyPorts. These connections are represented by csp_ports, which is a list of
    CSP ports corresponding to the connected PyPorts.
    """

    @property
    @abstractmethod
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """
        Abstract property to get a list of the corresponding CSP Ports of all
        connected PyPorts. The CSP Port is the low level interface of the
        backend messaging infrastructure which is used to send and receive data.

        Returns
        -------
        A list of all CSP Ports connected to the PyPort.
        """


class AbstractPyIOPort(AbstractPyPort):
    """Abstract class of an input/output Port implemented in python.

    A PyIOPort can either be an input or an output Port and is the common
    abstraction of PyInPort/PyOutPort.
    _csp_ports is a list of CSP Ports which are used to send/receive data by
    connected PyIOPorts.

    Parameters
    ----------
    csp_ports : list
        A list of CSP Ports used by this IO Port.

    process_model : AbstractProcessModel
        The process model used by the process of the Port.

    shape : tuple
        The shape of the Port.

    d_type: type
        The data type of the Port.

    Attributes
    ----------
    _csp_ports : list
        A list of CSP Ports used by this IO Port.
    """

    def __init__(self,
                 csp_ports: ty.List[AbstractCspPort],
                 process_model: AbstractProcessModel,
                 shape: ty.Tuple[int, ...],
                 d_type: type):
        self._csp_ports = csp_ports
        super().__init__(process_model, shape, d_type)

    @property
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Property to get the corresponding CSP Ports of all connected
        PyPorts (csp_ports). The CSP Port is the low level interface of the
        backend messaging infrastructure which is used to send and receive data.

        Returns
        -------
        A list of all CSP Ports connected to the PyPort.
        """
        return self._csp_ports


class AbstractTransformer(ABC):
    """Interface for Transformers that are used in receiving PyPorts to
    transform data."""

    @abstractmethod
    def transform(self,
                  data: np.ndarray,
                  csp_port: AbstractCspPort) -> np.ndarray:
        """Transforms incoming data in way that is determined by which CSP
        port the data is received.

        Parameters
        ----------
        data : numpy.ndarray
            data that will be transformed
        csp_port : AbstractCspPort
            CSP port that the data was received on

        Returns
        -------
        transformed_data : numpy.ndarray
            the transformed data
        """


class IdentityTransformer(AbstractTransformer):
    """Transformer that does not transform the data but returns it unchanged."""

    def transform(self,
                  data: np.ndarray,
                  _: AbstractCspPort) -> np.ndarray:
        return data


class VirtualPortTransformer(AbstractTransformer):
    def __init__(self,
                 csp_ports: ty.Dict[str, AbstractCspPort],
                 transform_funcs: ty.Dict[str, ty.List[ft.partial]]):
        """Transformer that implements the virtual ports on the path to the
        receiving PyPort.

        Parameters
        ----------
        csp_ports : ty.Dict[str, AbstractCspPort]
            mapping from a port ID to a CSP port
        transform_funcs : ty.Dict[str, ty.List[functools.partial]]
            mapping from a port ID to a list of function pointers that
            implement the behavior of the virtual pots on the path to the
            receiving PyPort.
        """
        self._csp_port_to_fp = {}

        if not csp_ports:
            raise AssertionError("'csp_ports' should not be empty.")

        # Associate CSP ports with the list of function pointers that must be
        # applied for the transformation.
        for port_id, csp_port in csp_ports.items():
            self._csp_port_to_fp[csp_port] = transform_funcs[port_id] \
                if port_id in transform_funcs else [lambda x: x]

    def transform(self,
                  data: np.ndarray,
                  csp_port: AbstractCspPort) -> np.ndarray:
        return self._get_transform(csp_port)(data)

    def _get_transform(self,
                       csp_port: AbstractCspPort) -> ty.Callable[[np.ndarray],
                                                                 np.ndarray]:
        """For a given CSP port, returns a function that applies, in sequence,
        all the function pointers associated with the incoming virtual
        ports.

        Example:
        Let the current PyPort be called C. It receives input from
        PyPorts A and B, and the connection from A to C goes through a
        sequence of virtual ports V1, V2, V3. Within PyPort C, there is a CSP
        port 'csp_port_a', that receives data from a CSP port in PyPort A.
        Then, the following call
        >>> csp_port_a : AbstractCspPort
        >>> data : np.ndarray
        >>> self._get_transform(csp_port_a)(data)
        takes the data 'data' and applies the function pointers associated
        with V1, V2, and V3.

        Parameters
        ----------
        csp_port : AbstractCspPort
            the CSP port on which the data is received, which is supposed
            to be transformed

        Returns
        -------
        transformation_function : ty.Callable
            function that transforms a given numpy array, e.g. by calling the
            returned function f(data)
        """
        if csp_port not in self._csp_port_to_fp:
            raise AssertionError(f"The CSP port '{csp_port.name}' is not "
                                 f"registered with a transformation function.")

        return ft.reduce(
            lambda f, g: lambda data: g(f(data)),
            self._csp_port_to_fp[csp_port],
            lambda h: h
        )


class PyInPort(AbstractPyIOPort):
    """Python implementation of InPort used within AbstractPyProcessModel.

    PyInPort is an input Port that can be used in a Process to receive data sent
    from a connected PyOutPort of another Process over a channel. PyInPort can
    receive (recv()) the data, which removes it from the channel, look (peek())
    at the data which keeps it on the channel or check (probe()) if there is
    data on the channel. The different class attributes are used to select the
    type of OutPorts via LavaPyType declarations in PyProcModels, e.g.,
    LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24) creates a PyInPort.
    A PyOutPort (source) can be connected to one or multiple PyInPorts (target).

    Parameters
    ----------
    csp_ports : ty.List[AbstractCspPort]
        Used to receive data from the referenced PyOutPort.

    process_model : AbstractProcessModel
        The process model used by the process of the Port.

    shape : tuple, default=tuple()
        The shape of the Port.

    d_type : type, default=int
        The data type of the Port.

    transformer : AbstractTransformer, default: identity function
        Enables transforming the received data in accordance with the
        virtual ports on the path to the PyInPort.

    Attributes
    ----------
    _transformer : AbstractTransformer
        Enables transforming the received data in accordance with the
        virtual ports on the path to the PyVarPort.

    VEC_DENSE : PyInPortVectorDense, default=None
        Type of PyInPort. CSP Port sends data as dense vector.

    VEC_SPARSE : PyInPortVectorSparse, default=None
        Type of PyInPort. CSP Port sends data as sparse vector (data + indices),
        so only entries which have changed in a vector need to be communicated.

    SCALAR_DENSE : PyInPortScalarDense, default=None
        Type of PyInPort. CSP Port sends data element by element for the whole
        data structure. So the CSP channel does need less memory to transfer
        data.

    SCALAR_SPARSE : PyInPortScalarSparse, default=None
        Type of PyInPort. CSP Port sends data element by element, but after each
        element the index of the data entry is also given. So only entries which
        need to be changed need to be communicated.
    """

    VEC_DENSE: ty.Type["PyInPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyInPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyInPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyInPortScalarSparse"] = None

    def __init__(
            self,
            csp_ports: ty.List[AbstractCspPort],
            process_model: AbstractProcessModel,
            shape: ty.Tuple[int, ...],
            d_type: type,
            transformer: ty.Optional[
                AbstractTransformer] = IdentityTransformer()
    ):
        self._transformer = transformer
        super().__init__(csp_ports, process_model, shape, d_type)

    @abstractmethod
    def recv(self):
        """Abstract method to receive data (vectors/scalars) sent from connected
        OutPorts (source Ports). Removes the retrieved data from the channel.
        Expects data on the channel and will block execution if there is no data
        to retrieve on the channel.

        Returns
        -------
        The scalar or vector received from a connected OutPort. If the InPort is
        connected to several OutPorts, their input is added in a point-wise
        fashion.
        """
        pass

    @abstractmethod
    def peek(self):
        """Abstract method to receive data (vectors/scalars) sent from connected
        OutPorts (source Ports). Keeps the data on the channel.

        Returns
        -------
        The scalar or vector received from a connected OutPort. If the InPort is
        connected to several OutPorts, their input is added in a point-wise
        fashion.
        """
        pass

    def probe(self) -> bool:
        """Method to check (probe) if there is data (vectors/scalars)
        to receive from connected OutPorts (source Ports).

        Returns
        -------
        result : bool
             Returns True only when there is data to receive from all connected
             OutPort channels.

        """
        return ft.reduce(
            lambda acc, csp_port: acc and csp_port.probe(),
            self.csp_ports,
            True,
        )


class PyInPortVectorDense(PyInPort):
    """Python implementation of PyInPort for dense vector data."""

    def recv(self) -> np.ndarray:
        """Method to receive data (vectors/scalars) sent from connected
        OutPorts (source Ports). Removes the retrieved data from the channel.
        Expects data on the channel and will block execution if there is no data
        to retrieve on the channel.

        Returns
        -------
        result : ndarray of shape _shape
            The vector received from a connected OutPort. If the InPort is
            connected to several OutPorts, their input is added in a point-wise
            fashion.
        """
        return ft.reduce(
            lambda acc, port: acc + self._transformer.transform(port.recv(),
                                                                port),
            self.csp_ports,
            np.zeros(self._shape, self._d_type)
        )

    def peek(self) -> np.ndarray:
        """Method to receive data (vectors) sent from connected
        OutPorts (source Ports). Keeps the data on the channel.

        Returns
        -------
        result : ndarray of shape _shape
            The vector received from a connected OutPort. If the InPort is
            connected to several OutPorts, their input is added in a point-wise
            fashion.
        """
        return ft.reduce(
            lambda acc, port: acc + self._transformer.transform(port.recv(),
                                                                port),
            self.csp_ports,
            np.zeros(self._shape, self._d_type),
        )


class PyInPortVectorSparse(PyInPort):
    """Python implementation of PyInPort for sparse vector data."""

    def recv(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """TBD"""
        csp_port = self._csp_ports[0]
        length = csp_port.recv().flatten()[0]
        data = csp_port.recv().flatten()[:length]
        index = csp_port.recv().flatten()[:length]
        return data, index

    def peek(self) -> ty.Tuple[np.ndarray, np.ndarray]:
        """TBD"""
        pass


class PyInPortScalarDense(PyInPort):
    """Python implementation of PyInPort for dense scalar data."""

    def recv(self) -> int:
        """TBD"""
        pass

    def peek(self) -> int:
        """TBD"""
        pass


class PyInPortScalarSparse(PyInPort):
    """Python implementation of PyInPort for sparse scalar data."""

    def recv(self) -> ty.Tuple[int, int]:
        """TBD"""
        pass

    def peek(self) -> ty.Tuple[int, int]:
        """TBD"""
        pass


PyInPort.VEC_DENSE = PyInPortVectorDense
PyInPort.VEC_SPARSE = PyInPortVectorSparse
PyInPort.SCALAR_DENSE = PyInPortScalarDense
PyInPort.SCALAR_SPARSE = PyInPortScalarSparse


class PyOutPort(AbstractPyIOPort):
    """Python implementation of OutPort used within AbstractPyProcessModels.

    PyOutPort is an output Port sending data to a connected input Port
    (PyInPort) over a channel. PyOutPort can send (send()) the data by adding it
    to the channel, or it can clear (flush()) the channel to remove any data
    from it. The different class attributes are used to select the type of
    OutPorts via LavaPyType declarations in PyProcModels, e.g., LavaPyType(
    PyOutPort.VEC_DENSE, np.int32, precision=24) creates a PyOutPort.
    A PyOutPort (source) can be connected to one or multiple PyInPorts (target).

    Parameters
    ----------
    csp_ports : list
        A list of CSP Ports used by this IO Port.
    process_model : AbstractProcessModel
        The process model used by the process of the Port.
    shape : tuple
        The shape of the Port.
    d_type: type
        The data type of the Port.

    Attributes
    ----------------
    VEC_DENSE : PyOutPortVectorDense, default=None
        Type of PyInPort. CSP Port sends data as dense vector.
    VEC_SPARSE : PyOutPortVectorSparse, default=None
        Type of PyInPort. CSP Port sends data as sparse vector (data + indices),
        so only entries which have changed in a vector need to be communicated.
    SCALAR_DENSE : PyOutPortScalarDense, default=None
        Type of PyInPort. CSP Port sends data element by element for the whole
        data structure. So the CSP channel does need less memory to transfer
        data.
    SCALAR_SPARSE : PyOutPortScalarSparse, default=None
        Type of PyInPort. CSP Port sends data element by element, but after each
        element the index of the data entry is also given. So only entries which
        need to be changed need to be communicated.
    """

    VEC_DENSE: ty.Type["PyOutPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyOutPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyOutPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyOutPortScalarSparse"] = None

    @abstractmethod
    def send(self, data: ty.Union[np.ndarray, int]):
        """Abstract method to send data to the connected Port PyInPort (target).

        Parameters
        ----------
        data : ndarray or int
            The data (vector or scalar) to be sent to the InPort (target).
        """
        pass

    def flush(self):
        """TBD"""
        pass

    def advance_to_time_step(self, ts: int):
        for csp_port in self.csp_ports:
            if hasattr(csp_port, "advance_to_time_step"):
                csp_port.advance_to_time_step(ts)


class PyOutPortVectorDense(PyOutPort):
    """Python implementation of PyOutPort for dense vector data."""

    def send(self, data: np.ndarray):
        """Abstract method to send data to the connected in Port (target).

        Sends data only if the OutPort is connected to at least one InPort.

        Parameters
        ----------
        data : ndarray
            The data vector to be sent to the in Port (target).
        """
        for csp_port in self.csp_ports:
            csp_port.send(data)


class PyOutPortVectorSparse(PyOutPort):
    """Python implementation of PyOutPort for sparse vector data."""

    def send(self, data: np.ndarray, indices: np.ndarray):
        """TBD"""
        data_clone = np.copy(data)
        indices_clone = np.copy(indices)
        data_length: np.ndarray = np.array([len(data.flatten())],
                                           dtype=np.int32)
        for csp_port in self.csp_ports:
            if csp_port.is_msg_size_static():
                data_length.resize(csp_port.shape)
                data_clone.resize(csp_port.shape)
                indices_clone.resize(csp_port.shape)
                csp_port.send(data_length)
                csp_port.send(data_clone)
                csp_port.send(indices_clone)
            else:
                csp_port.send(
                    np.concatenate(arrays=[data_length,
                                           data_clone,
                                           indices_clone],
                                   dtype=np.int32))


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
    """Helper class to specify constants. Used for communication between
    PyRefPorts and PyVarPorts."""
    GET = enum_to_np(0)
    SET = enum_to_np(1)


class PyRefPort(AbstractPyPort):
    """Python implementation of RefPort used within AbstractPyProcessModels.

    A PyRefPort is a Port connected to a VarPort of a variable Var of another
    Process. It is used to get or set the value of the referenced Var across
    Processes. A PyRefPort is connected via two CSP channels and corresponding
    CSP ports to a PyVarPort. One channel is used to send data from the
    PyRefPort to the PyVarPort and the other channel is used to receive data
    from the PyVarPort. PyRefPorts can get the value of a referenced Var
    (read()), set the value of a referenced Var (write()) and block execution
    until receipt of prior 'write' commands (sent from PyRefPort to PyVarPort)
    have been acknowledged (wait()).

    Parameters
    ----------
    csp_send_port : CspSendPort or None
        Used to send data to the referenced Port PyVarPort (target).

    csp_recv_port: CspRecvPort or None
        Used to receive data from the referenced Port PyVarPort (source).

    process_model : AbstractProcessModel
        The process model used by the process of the Port.

    shape : tuple, default=tuple()
        The shape of the Port.

    d_type : type, default=int
        The data type of the Port.

    transformer : AbstractTransformer, default: identity function
        Enables transforming the received data in accordance with the
        virtual ports on the path to the PyRefPort.

    Attributes
    ----------
    _csp_send_port : CspSendPort
        Used to send data to the referenced Port PyVarPort (target).
    _csp_recv_port : CspRecvPort
        Used to receive data from the referenced Port PyVarPort (source).
    _transformer : AbstractTransformer
        Enables transforming the received data in accordance with the
        virtual ports on the path to the PyRefPort.

    VEC_DENSE : PyRefPortVectorDense, default=None
        Type of PyInPort. CSP Port sends data as dense vector.
    VEC_SPARSE : PyRefPortVectorSparse, default=None
        Type of PyInPort. CSP Port sends data as sparse vector (data + indices),
        so only entries which have changed in a vector need to be communicated.
    SCALAR_DENSE : PyRefPortScalarDense, default=None
        Type of PyInPort. CSP Port sends data element by element for the whole
        data structure. So the CSP channel does need less memory to transfer
        data.
    SCALAR_SPARSE : PyRefPortScalarSparse, default=None
        Type of PyInPort. CSP Port sends data element by element, but after each
        element the index of the data entry is also given. So only entries which
        need to be changed need to be communicated.
    """

    VEC_DENSE: ty.Type["PyRefPortVectorDense"] = None
    VEC_SPARSE: ty.Type["PyRefPortVectorSparse"] = None
    SCALAR_DENSE: ty.Type["PyRefPortScalarDense"] = None
    SCALAR_SPARSE: ty.Type["PyRefPortScalarSparse"] = None

    def __init__(
            self,
            csp_send_port: ty.Optional[CspSendPort],
            csp_recv_port: ty.Optional[CspRecvPort],
            process_model: AbstractProcessModel,
            shape: ty.Tuple[int, ...] = tuple(),
            d_type: type = int,
            transformer: ty.Optional[
                AbstractTransformer] = IdentityTransformer()
    ):
        self._transformer = transformer
        self._csp_recv_port = csp_recv_port
        self._csp_send_port = csp_send_port
        super().__init__(process_model, shape, d_type)

    @property
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Property to get the corresponding CSP Ports of all connected
        PyPorts (csp_ports). The CSP Port is the low level interface of the
        backend messaging infrastructure which is used to send and receive data.

        Returns
        -------
        A list of all CSP Ports connected to the PyPort.
        """
        if self._csp_send_port is not None and self._csp_recv_port is not None:
            return [self._csp_send_port, self._csp_recv_port]
        else:
            # In this case the Port was not connected
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

    def wait(self):
        """Blocks execution until receipt of prior 'write' commands (sent from
         RefPort to VarPort) have been acknowledged. Calling wait() ensures that
         the value written by the RefPort can be received (and set) by the
         VarPort at the same time step. If wait() is not called, it is possible
         that the value is received only at the next time step
         (non-deterministic).

         >>> port = PyRefPort()
         >>> port.write(5)
         >>> # potentially do other stuff
         >>> port.wait()  # waits until (all) previous writes have finished

         Preliminary implementation. Currently, a simple read() ensures the
         writes have been acknowledged. This is inefficient and will be
         optimized later at the CspChannel level"""
        self.read()


class PyRefPortVectorDense(PyRefPort):
    """Python implementation of RefPort for dense vector data."""

    def read(self) -> np.ndarray:
        """Method to request and return data from a referenced Var using a
        PyVarPort.

        Returns
        -------
        result : ndarray of shape _shape
            The value of the referenced Var.
        """
        if self._csp_send_port and self._csp_recv_port:
            if not hasattr(self, 'get_header'):
                self.get_header = (np.ones(self._csp_send_port.shape)
                                   * VarPortCmd.GET)
            self._csp_send_port.send(self.get_header)
            return self._transformer.transform(self._csp_recv_port.recv(),
                                               self._csp_recv_port)
        else:
            if not hasattr(self, 'get_zeros'):
                self.get_zeros = np.zeros(self._shape, self._d_type)
            return self.get_zeros

    def write(self, data: np.ndarray):
        """Abstract method to write data to a VarPort to set the value of the
        referenced Var.

        Parameters
        ----------
        data : ndarray
            The data to send via _csp_send_port.
        """
        if self._csp_send_port:
            if not hasattr(self, 'set_header'):
                self.set_header = (np.ones(self._csp_send_port.shape)
                                   * VarPortCmd.SET)
            self._csp_send_port.send(self.set_header)
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

    A PyVarPort is a Port linked to a variable Var of a Process and might be
    connected to a RefPort of another process. It is used to get or set the
    value of the referenced Var across Processes. A PyVarPort is connected via
    two channels to a PyRefPort. One channel is used to send data from the
    PyRefPort to the PyVarPort and the other is used to receive data from the
    PyVarPort. PyVarPorts set or send the value of the linked Var (service())
    given the command VarPortCmd received by a connected PyRefPort.

    Parameters
    ----------
    var_name : str
        The name of the Var linked to this VarPort.

    csp_send_port : CspSendPort or None
        Csp Port used to send data to the referenced in Port (target).

    csp_recv_port: CspRecvPort or None
        Csp Port used to receive data from the referenced Port (source).

    process_model : AbstractProcessModel
        The process model used by the process of the Port.

    shape : tuple, default=tuple()
        The shape of the Port.

    d_type: type, default=int
        The data type of the Port.

    transformer : AbstractTransformer, default: identity function
        Enables transforming the received data in accordance with the
        virtual ports on the path to the PyVarPort.

    Attributes
    ----------
    var_name : str
        The name of the Var linked to this VarPort.
    _csp_send_port : CspSendPort
        Used to send data to the referenced Port PyRefPort (target).
    _csp_recv_port : CspRecvPort
        Used to receive data from the referenced Port PyRefPort (source).
    _transformer : AbstractTransformer
        Enables transforming the received data in accordance with the
        virtual ports on the path to the PyVarPort.

    VEC_DENSE : PyVarPortVectorDense, default=None
       Type of PyInPort. CSP Port sends data as dense vector.
    VEC_SPARSE : PyVarPortVectorSparse, default=None
        Type of PyInPort. CSP Port sends data as sparse vector (data + indices),
        so only entries which have changed in a vector need to be communicated.
    SCALAR_DENSE : PyVarPortScalarDense, default=None
        Type of PyInPort. CSP Port sends data element by element for the whole
        data structure. So the CSP channel does need less memory to transfer
        data.
    SCALAR_SPARSE : PyVarPortScalarSparse, default=None
        Type of PyInPort. CSP Port sends data element by element, but after each
        element the index of the data entry is also given. So only entries which
        need to be changed need to be communicated.
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
                 d_type: type = int,
                 transformer: AbstractTransformer = IdentityTransformer()):

        self._transformer = transformer
        self._csp_recv_port = csp_recv_port
        self._csp_send_port = csp_send_port
        self.var_name = var_name
        super().__init__(process_model, shape, d_type)

    @property
    def csp_ports(self) -> ty.List[AbstractCspPort]:
        """Property to get the corresponding CSP Ports of all connected
        PyPorts (csp_ports). The CSP Port is the low level interface of the
        backend messaging infrastructure which is used to send and receive data.

        Returns
        -------
        A list of all CSP Ports connected to the PyPort.
        """
        if self._csp_send_port is not None and self._csp_recv_port is not None:
            return [self._csp_send_port, self._csp_recv_port]
        else:
            # In this case the Port was not connected
            return []

    @abstractmethod
    def service(self):
        """Abstract method to set the value of the linked Var of the VarPort,
        received from the connected RefPort, or to send the value of the linked
        Var of the VarPort to the connected RefPort. The connected RefPort
        determines whether it will perform a read() or write() operation by
        sending a command VarPortCmd.
        """
        pass


class PyVarPortVectorDense(PyVarPort):
    """Python implementation of VarPort for dense vector data."""

    def service(self):
        """Method to set the value of the linked Var of the VarPort,
        received from the connected RefPort, or to send the value of the linked
        Var of the VarPort to the connected RefPort. The connected RefPort
        determines whether it will perform a read() or write() operation by
        sending a command VarPortCmd.
        """

        # Inspect incoming data
        if self._csp_send_port is not None and self._csp_recv_port is not None:
            if self._csp_recv_port.probe():
                # If received data is a matrix, flatten and take the first
                # element as cmd
                cmd = enum_to_np((self._csp_recv_port.recv()).flatten()[0])

                # Set the value of the Var with the given data
                if enum_equal(cmd, VarPortCmd.SET):
                    data = self._transformer.transform(
                        self._csp_recv_port.recv(),
                        self._csp_recv_port)
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

    def service(self):
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

    def service(self):
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

    def service(self):
        """TBD"""
        pass


PyVarPort.VEC_DENSE = PyVarPortVectorDense
PyVarPort.VEC_SPARSE = PyVarPortVectorSparse
PyVarPort.SCALAR_DENSE = PyVarPortScalarDense
PyVarPort.SCALAR_SPARSE = PyVarPortScalarSparse


class RefVarTypeMapping:
    """Class to get the mapping of PyRefPort types to PyVarPort types.

    PyRefPorts and PyVarPorts can be implemented as different subtypes, defining
    the format of the data to process. To connect PyRefPorts and PyVarPorts they
    need to have a compatible data format.
    This class maps the fitting data format between PyRefPorts and PyVarPorts.

    Attributes
    ----------------
    mapping : dict
        Dictionary containing the mapping of compatible PyRefPort types to
        PyVarPort types.

    """

    mapping: ty.Dict[ty.Type[PyRefPort], ty.Type[PyVarPort]] = {
        PyRefPortVectorDense: PyVarPortVectorDense,
        PyRefPortVectorSparse: PyVarPortVectorSparse,
        PyRefPortScalarDense: PyVarPortScalarDense,
        PyRefPortScalarSparse: PyVarPortScalarSparse}

    @classmethod
    def get(cls, ref_port: ty.Type[PyRefPort]) -> ty.Type[PyVarPort]:
        """Class method to return the compatible PyVarPort type given the
        PyRefPort type.

        Parameters
        ----------
        ref_port : ty.Type[PyRefPort]
            PyRefPort type to be mapped to a PyVarPort type.

        Returns
        -------
        result : ty.Type[PyVarPort]
            PyVarPort type compatible to given PyRefPort type.

        """
        return cls.mapping[ref_port]
