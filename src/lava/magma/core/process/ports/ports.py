# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from __future__ import annotations
import typing as ty
from abc import ABC, abstractmethod
import math
import numpy as np
import functools as ft

from lava.magma.core.process.interfaces import AbstractProcessMember
import lava.magma.core.process.ports.exceptions as pe
from lava.magma.core.process.ports.connection_config import ConnectionConfig
from lava.magma.core.process.ports.reduce_ops import AbstractReduceOp
from lava.magma.core.process.variable import Var

ConnectionConfigs = ty.Union["ConnectionConfig", ty.List["ConnectionConfig"]]


def to_list(obj: ty.Any) -> ty.List[ty.Any]:
    """If 'obj' is not a list, converts 'obj' into [obj]."""
    if not isinstance(obj, list):
        obj = [obj]
    return obj


def is_disjoint(a: ty.List, b: ty.List):
    """Checks that both lists are disjoint."""
    return set(a).isdisjoint(set(b))


def create_port_id(proc_id: int, port_name: str) -> str:
    """Generates a string-based ID for a port that makes it identifiable
    within a network of Processes.

    Parameters
    ----------
    proc_id : int
        ID of the Process that the Port is associated with
    port_name : str
        name of the Port

    Returns
    -------
    port_id : str
        ID of a port
    """
    return str(proc_id) + "_" + port_name


class AbstractPort(AbstractProcessMember):
    """Abstract base class for any type of port of a Lava Process.

    Ports of a process can be connected to ports of other processes to enable
    message-based communication via channels. Sub classes of AbstractPort
    only facilitate connecting to other compatible ports. Message-passing
    itself is only handled after compilation at runtime by port
    implementations within the corresponding ProcessModel.

    Ports are tensor-valued, have a name and a parent process. In addition,
    a port may have zero or more input and output connections that contain
    references to ports that connect to this port or that this port connects
    to. Port to port connections are directional and connecting ports,
    effectively means to associate them with each other as inputs or outputs.
    These connections, imply an a-cyclic graph structure that allows the
    compiler to infer connections between processes.

    Parameters
    ----------
    shape: tuple[int, ...]
        Determines the number of connections created by this port
    """

    def __init__(self, shape: ty.Tuple[int, ...]):
        super().__init__(shape)
        self.in_connections: ty.List[AbstractPort] = []
        self.out_connections: ty.List[AbstractPort] = []
        self.connection_configs: ty.Dict[AbstractPort, ConnectionConfig] = {}

    def _validate_ports(
            self,
            ports: ty.List["AbstractPort"],
            port_type: ty.Type["AbstractPort"],
            assert_same_shape: bool = True,
            assert_same_type: bool = False,
    ):
        """Checks that each port in 'ports' is of type 'port_type' and that
        shapes of each port is identical to this port's shape."""
        cls_name = port_type.__name__
        specific_cls = ports[0].__class__
        for p in ports:
            if not isinstance(p, port_type):
                raise AssertionError("'ports' must be of type {} but "
                                     "found {}.".format(cls_name, p.__class__))
            if assert_same_type:
                if not isinstance(p, specific_cls):
                    raise AssertionError(
                        "All ports must be of same type but found {} "
                        "and {}.".format(specific_cls, p.__class__)
                    )
            if assert_same_shape:
                if self.shape != p.shape:
                    raise AssertionError("Shapes {} and {} "
                                         "are incompatible."
                                         .format(self.shape, p.shape))

    def _add_inputs(self, inputs: ty.List["AbstractPort"]):
        """Adds new input connections to port. Does not allow that same
        inputs get connected more than once to port."""
        if not is_disjoint(self.in_connections, inputs):
            raise pe.DuplicateConnectionError()
        self.in_connections += inputs

    def _add_outputs(self, outputs: ty.List["AbstractPort"]):
        """Adds new output connections to port. Does not allow that same
        outputs get connected more than once to port."""
        if not is_disjoint(self.out_connections, outputs):
            raise pe.DuplicateConnectionError()
        self.out_connections += outputs

    def _connect_forward(
            self,
            ports: ty.List["AbstractPort"],
            port_type: ty.Type["AbstractPort"],
            connection_configs: ty.List[ty.Optional[ConnectionConfig]],
            assert_same_shape: bool = True,
            assert_same_type: bool = True):
        """Creates a forward connection from this AbstractPort to other
        ports by adding other ports to this AbstractPort's out_connection and
        by adding this AbstractIOPort to other port's in_connections."""

        self._validate_ports(
            ports, port_type, assert_same_shape, assert_same_type
        )
        # Add other ports to this port's output connections
        self._add_outputs(ports)
        # Add this port to input connections of other ports
        if len(connection_configs) == 1 and len(ports) > 1:
            connection_configs = connection_configs * len(ports)
        for p, connection_config in zip(ports, connection_configs):
            p._add_inputs([self])
            if connection_config:
                self.connection_configs[p] = connection_config
                p.connection_configs[self] = connection_config

    def _connect_backward(
            self,
            ports: ty.List["AbstractPort"],
            port_type: ty.Type["AbstractPort"],
            connection_configs: ty.List[ty.Optional[ConnectionConfig]],
            assert_same_shape: bool = True,
            assert_same_type: bool = True):
        """Creates a backward connection from other ports to this
        AbstractPort by adding other ports to this AbstractPort's
        in_connection and by adding this AbstractPort to other port's
        out_connections."""

        self._validate_ports(
            ports, port_type, assert_same_shape, assert_same_type
        )
        # Add other ports to this port's input connections
        self._add_inputs(ports)
        # Add this port to output connections of other ports
        if len(connection_configs) == 1 and len(ports) > 1:
            connection_configs = connection_configs * len(ports)
        for p, connection_config in zip(ports, connection_configs):
            p._add_outputs([self])
            if connection_config:
                p.connection_configs[self] = connection_config
                self.connection_configs[p] = connection_config

    def get_src_ports(self, _include_self=False) -> ty.List["AbstractSrcPort"]:
        """Returns the list of all source ports that connect either directly
        or indirectly (through other ports) to this port."""
        if len(self.in_connections) == 0:
            if _include_self:
                return [ty.cast(AbstractSrcPort, self)]
            else:
                return []
        else:
            ports = []
            for p in self.in_connections:
                ports += p.get_src_ports(True)
            return ports

    def get_incoming_transform_funcs(self) -> ty.Dict[str, ty.List[ft.partial]]:
        """Returns the incoming transformation functions for all incoming
        connections.

        Returns
        -------
        dict(list(functools.partial))
            A dictionary that maps from the ID of an incoming source port to
            the list of transformation functions implementing the virtual
            ports on the way to the current port. The transformation
            functions in the list are sorted from source to destination port.
        """
        transform_func_map = {}
        for port in self.in_connections:
            src_port_id, vps = port.get_incoming_virtual_ports()
            transform_func_list = [vp.get_transform_func_fwd() for vp in vps]
            if transform_func_list:
                transform_func_map[src_port_id] = transform_func_list

        return transform_func_map

    def get_incoming_virtual_ports(self) \
            -> ty.Tuple[str, ty.List["AbstractVirtualPort"]]:
        """Returns the list of all incoming virtual ports in order from
        source to the current port.

        Returns
        -------
        tuple(str, list(AbstractVirtualPorts))
            The string of the tuple is the ID of the source port, the list
            is the list of all incoming virtual ports, sorted from source to
            destination port.
        """
        if len(self.in_connections) == 0:
            src_port_id = create_port_id(self.process.id, self.name)
            return src_port_id, []
        else:
            virtual_ports = []
            src_port_id = None
            for p in self.in_connections:
                p_id, vps = p.get_incoming_virtual_ports()
                virtual_ports += vps
                if p_id:
                    src_port_id = p_id

            if isinstance(self, AbstractVirtualPort):
                if isinstance(self, ConcatPort):
                    raise NotImplementedError("ConcatPorts are not yet "
                                              "supported.")
                virtual_ports.append(self)

            return src_port_id, virtual_ports

    def get_outgoing_transform_funcs(self) -> ty.Dict[str, ty.List[ft.partial]]:
        """Returns the outgoing transformation functions for all outgoing
        connections.

        Returns
        -------
        dict(list(functools.partial))
            A dictionary that maps from the ID of a destination port to
            the list of transformation functions implementing the virtual
            ports on the way from the current port. The transformation
            functions in the list are sorted from source to destination port.
        """
        transform_funcs = {}
        for p in self.out_connections:
            dst_port_id, vps = p.get_outgoing_virtual_ports()
            transform_funcs[dst_port_id] = \
                [vp.get_transform_func_bwd() for vp in vps]
        return transform_funcs

    def get_outgoing_virtual_ports(self) \
            -> ty.Tuple[str, ty.List["AbstractVirtualPort"]]:
        """Returns the list of all outgoing virtual ports in order from
        the current port to the destination port.

        Returns
        -------
        tuple(str, list(AbstractVirtualPorts))
            The string of the tuple is the ID of the destination port, the list
            is the list of all outgoing virtual ports, sorted from source to
            destination port.
        """
        if len(self.out_connections) == 0:
            dst_port_id = create_port_id(self.process.id, self.name)
            return dst_port_id, []
        else:
            virtual_ports = []
            dst_port_id = None
            for p in self.out_connections:
                p_id, vps = p.get_outgoing_virtual_ports()
                virtual_ports += vps
                if p_id:
                    dst_port_id = p_id

            if isinstance(self, AbstractVirtualPort):
                if isinstance(self, ConcatPort):
                    raise NotImplementedError("ConcatPorts are not yet "
                                              "supported.")
                virtual_ports.append(self)

            return dst_port_id, virtual_ports

    def get_dst_ports(self, _include_self=False) -> ty.List["AbstractDstPort"]:
        """Returns the list of all destination ports that this port connects to
        either directly or indirectly (through other ports)."""
        if len(self.out_connections) == 0:
            if _include_self:
                return [ty.cast(AbstractDstPort, self)]
            else:
                return []
        else:
            ports = []
            for p in self.out_connections:
                ports += p.get_dst_ports(True)
            return ports

    def reshape(self, new_shape: ty.Tuple[int, ...]) -> "ReshapePort":
        """Reshapes this port by deriving and returning a new virtual
        ReshapePort with the new shape. This implies that the resulting
        ReshapePort can only be forward connected to another port.

        Parameters
        ----------
        new_shape: tuple[int, ...]
            New shape of port. Number of total elements must not change.
        """
        if self.size != math.prod(new_shape):
            raise pe.ReshapeError(self.shape, new_shape)

        reshape_port = ReshapePort(new_shape, old_shape=self.shape)
        self._connect_forward(
            [reshape_port], AbstractPort, [None], assert_same_shape=False
        )
        return reshape_port

    def flatten(self) -> "ReshapePort":
        """Flattens this port to a (N,)-shaped port by deriving and returning
        a new virtual ReshapePort with a N equal to the total number of
        elements of this port."""
        return self.reshape((self.size,))

    def concat_with(
            self,
            ports: ty.Union["AbstractPort", ty.List["AbstractPort"]],
            axis: int,
    ) -> "ConcatPort":
        """Concatenates this port with other ports in given order along given
        axis by deriving and returning a new virtual ConcatPort. This implies
        resulting ConcatPort can only be forward connected to another port.
        All ports must have the same shape outside of the concatenation
        dimension.

        Parameters
        ----------
        ports: ty.Union["AbstractPort", ty.List["AbstractPort"]]
            Port(s) that will be concatenated after this port.
        axis: int
            Axis/dimension along which ports are concatenated.
        """
        ports = [self] + to_list(ports)
        if isinstance(self, AbstractIOPort):
            port_type = AbstractIOPort
        else:
            port_type = AbstractRVPort
        self._validate_ports(ports, port_type, assert_same_shape=False)
        return ConcatPort(ports, axis)

    def transpose(
        self,
        axes: ty.Optional[ty.Union[ty.Tuple[int, ...],
                                   ty.List]] = None
    ) -> "TransposePort":
        """Permutes the tensor dimension of this port by deriving and returning
        a new virtual TransposePort the new permuted dimension. This implies
        that the resulting TransposePort can only be forward connected to
        another port.

        Parameters
        ----------
        axes: ty.Optional[ty.Union[ty.Tuple[int, ...], ty.List]]
            Order of permutation. Number of total elements and number of
            dimensions must not change.
        """
        if axes is None:
            axes = tuple(reversed(range(len(self.shape))))
        else:
            if len(self.shape) != len(axes):
                raise pe.TransposeShapeError(self.shape, axes)

            # Check that none of the given axes are out of bounds for the
            # shape of the parent port.
            for idx in axes:
                # Compute the positive index irrespective of the sign of 'idx'
                idx_positive = len(self.shape) + idx if idx < 0 else idx
                # Make sure the positive index is not out of bounds
                if idx_positive < 0 or idx_positive >= len(self.shape):
                    raise pe.TransposeIndexError(self.shape, axes, idx)

        new_shape = tuple([self.shape[i] for i in axes])
        transpose_port = TransposePort(new_shape, axes)
        self._connect_forward(
            [transpose_port], AbstractPort, [None], assert_same_shape=False
        )
        return transpose_port


class AbstractIOPort(AbstractPort):
    """Abstract base class for InPorts and OutPorts.
    This class needs no implementation and only serves to establish a clear
    type hierarchy needed for validating connections.
    """


class AbstractRVPort(AbstractPort):
    """Abstract base class for RefPorts and VarPorts.
    This class needs no implementation and only serves to establish a clear
    type hierarchy needed for validating connections.
    """


class AbstractSrcPort(ABC):
    """Interface for source ports such as OutPorts and RefPorts from which
    connections originate.
    This class needs no implementation and only serves to establish a clear
    type hierarchy needed for validating connections.
    """


class AbstractDstPort(ABC):
    """Interface for destination ports such as InPorts and VarPorts in which
    connections terminate.
    This class needs no implementation and only serves to establish a clear
    type hierarchy needed for validating connections.
    """


class OutPort(AbstractIOPort, AbstractSrcPort):
    """Output ports are members of a Lava Process and can be connected to
    other ports to facilitate sending of messages via channels.

    OutPorts connect to other InPorts of peer processes or to other OutPorts of
    processes that contain this OutPort's parent process as a sub process.
    Similarly, OutPorts can receive connections from other OutPorts of nested
    sub processes.
    """

    def __init__(self, shape: ty.Tuple[int, ...]):
        super().__init__(shape)
        self.external_pipe_flag = False
        self.external_pipe_buffer_size = 64

    def flag_external_pipe(self, buffer_size=None):
        self.external_pipe_flag = True

        if buffer_size is not None:
            self.external_pipe_buffer_size = buffer_size

    def connect(
            self,
            ports: ty.Union["AbstractIOPort", ty.List["AbstractIOPort"]],
            connection_configs: ty.Optional[ConnectionConfigs] = None):
        """Connects this OutPort to other InPort(s) of another process
        or to OutPort(s) of its parent process.

        Parameters
        ----------
        ports: ty.Union["AbstractIOPort", ty.List["AbstractIOPort"]]
            The AbstractIOPort(s) to connect to.
        connection_configs: ConnectionConfigs
            Configuration for this connection. See "ConnectionConfig" class.
        """
        self._connect_forward(to_list(ports),
                              AbstractIOPort,
                              to_list(connection_configs))

    def connect_from(self,
                     ports: ty.Union["OutPort", ty.List["OutPort"]],
                     connection_configs: ty.Optional[ConnectionConfigs] = None):
        """Connects other OutPort(s) of a nested process to this OutPort.
        OutPorts cannot receive connections from other InPorts.

        Parameters
        ----------
        ports: ty.Union["OutPort", ty.List["OutPort"]]
            The OutPorts(s) that connect to this OutPort.
        connection_configs: ConnectionConfigs
            Configuration for this connection. See "ConnectionConfig" class.
        """
        self._connect_backward(to_list(ports),
                               OutPort,
                               to_list(connection_configs))


class InPort(AbstractIOPort, AbstractDstPort):
    """Input ports are members of a Lava Process and can be connected to
    other ports to facilitate receiving of messages via channels.

    InPorts can receive connections from other OutPorts of peer processes
    or from other InPorts of processes that contain this InPort's parent
    process as a sub process. Similarly, InPorts can connect to other InPorts
    of nested sub processes.

    Parameters
    ----------
    shape: tuple[int, ...]
        Determines the number of connections created by this port.
    reduce_op: ty.Optional[ty.Type[AbstractReduceOp]]
        Operation to be applied on incoming data, default: None.
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            reduce_op: ty.Optional[ty.Type[AbstractReduceOp]] = None,
    ):
        super().__init__(shape)
        self._reduce_op = reduce_op

        self.external_pipe_flag = False
        self.external_pipe_buffer_size = 64

    def flag_external_pipe(self, buffer_size=None):
        self.external_pipe_flag = True

        if buffer_size is not None:
            self.external_pipe_buffer_size = buffer_size

    def connect(self,
                ports: ty.Union["InPort", ty.List["InPort"]],
                connection_configs: ty.Optional[ConnectionConfigs] = None):
        """Connects this InPort to other InPort(s) of a nested process. InPorts
        cannot connect to other OutPorts.

        Parameters
        ----------
        ports: ty.Union["InPort", ty.List["InPort"]]
            The InPort(s) to connect to.
        connection_configs: ConnectionConfigs
            Configuration for this connection. See "ConnectionConfig" class.
        """
        self._connect_forward(to_list(ports),
                              InPort,
                              to_list(connection_configs))

    def connect_from(
            self, ports: ty.Union["AbstractIOPort", ty.List["AbstractIOPort"]],
            connection_configs: ty.Optional[ConnectionConfigs] = None):
        """Connects other OutPort(s) to this InPort or connects other
        InPort(s) of parent process to this InPort.

        Parameters
        ----------
        ports: ty.Union["AbstractIOPort", ty.List["AbstractIOPort"]]
            The AbstractIOPort(s) that connect to this InPort.
        connection_configs: ConnectionConfigs
            Configuration for this connection. See "ConnectionConfig" class.
        """
        self._connect_backward(to_list(ports),
                               AbstractIOPort,
                               to_list(connection_configs))


class RefPort(AbstractRVPort, AbstractSrcPort):
    """RefPorts are members of a Lava Process and can be connected to
    internal Lava Vars of other processes to facilitate direct shared memory
    access to those processes.

    Shared-memory-based communication can have side-effects and should
    therefore be used with caution.

    RefPorts connect to other VarPorts of peer processes or to other RefPorts
    of processes that contain this RefPort's parent process as a sub process
    via the connect(..) method..
    Similarly, RefPorts can receive connections from other RefPorts of nested
    sub processes via the connect_from(..) method.

    Here, VarPorts only serve as a wrapper for Vars. VarPorts can be created
    statically during process definition to explicitly expose a Var for
    remote memory access (which might be safer).
    Alternatively, VarPorts can be created dynamically by connecting a
    RefPort to a Var via the connect_var(..) method."""

    def connect(
            self, ports: ty.Union["AbstractRVPort", ty.List["AbstractRVPort"]],
            connection_configs: ty.Optional[ConnectionConfigs] = None):
        """Connects this RefPort to other VarPort(s) of another process
        or to RefPort(s) of its parent process.

        Parameters
        ----------
        ports: ty.Union["AbstractRVPort", ty.List["AbstractRVPort"]]
            The AbstractRVPort(s) to connect to.
        connection_configs: ConnectionConfigs
            Configuration for this connection. See "ConnectionConfig" class.
        """

        # Check if multiple ports should be connected (currently not supported)
        if len(to_list(ports)) > 1 \
                or (len(self.get_dst_ports()) > 0
                    and not isinstance(ports, AbstractSrcPort)) \
                or (len(self.get_src_ports()) > 0
                    and not isinstance(ports, AbstractDstPort)):
            raise AssertionError(
                "Currently only 1:1 connections are supported for RefPorts:"
                " {!r}: {!r}".format(
                    self.process.__class__.__name__, self.name))

        for p in to_list(ports):
            if not isinstance(p, RefPort) and not isinstance(p, VarPort):
                raise TypeError(
                    "RefPorts can only be connected to RefPorts or "
                    "VarPorts: {!r}: {!r} -> {!r}: {!r}  To connect a RefPort "
                    "to a Var, use <connect_var>".format(
                        self.process.__class__.__name__, self.name,
                        p.process.__class__.__name__, p.name))
        self._connect_forward(to_list(ports),
                              AbstractRVPort,
                              to_list(connection_configs))

    def connect_from(self, ports: ty.Union["RefPort", ty.List["RefPort"]],
                     connection_configs: ty.Optional[ConnectionConfigs] = None):
        """Connects other RefPort(s) of a nested process to this RefPort.
        RefPorts cannot receive connections from other VarPorts.

        Parameters
        ----------
        ports: ty.Union["RefPort", ty.List["RefPort"]]
            The RefPort(s) that connect to this RefPort.
        connection_configs: ConnectionConfigs
            Configuration for this connection. See "ConnectionConfig" class.
        """

        # Check if multiple ports should be connected (currently not supported)
        if len(to_list(ports)) > 1 \
                or (len(self.get_dst_ports()) > 0
                    and not isinstance(ports, AbstractSrcPort)) \
                or (len(self.get_src_ports()) > 0
                    and not isinstance(ports, AbstractDstPort)):
            raise AssertionError(
                "Currently only 1:1 connections are supported for RefPorts:"
                " {!r}: {!r}".format(
                    self.process.__class__.__name__, self.name))

        for p in to_list(ports):
            if not isinstance(p, RefPort):
                raise TypeError(
                    "RefPorts can only receive connections from RefPorts: "
                    "{!r}: {!r} -> {!r}: {!r}".format(
                        self.process.__class__.__name__, self.name,
                        p.process.__class__.__name__, p.name))
        self._connect_backward(to_list(ports),
                               RefPort,
                               to_list(connection_configs))

    def connect_var(self, variables: ty.Union[Var, ty.List[Var]],
                    connection_configs: ty.Optional[ConnectionConfigs] = None):
        """Connects this RefPort to Lava Process Var(s) to facilitate shared
        memory access.

        Parameters
        ----------
        variables: ty.Union[Var, ty.List[Var]]
            Var or list of Vars to connect to.
        connection_configs: ConnectionConfigs
            Configuration for this connection. See "ConnectionConfig" class.
        """

        # Check if multiple ports should be connected (currently not supported)
        if len(to_list(variables)) > 1 \
                or (len(self.get_dst_ports()) > 0
                    and not isinstance(variables, AbstractSrcPort)) \
                or (len(self.get_src_ports()) > 0
                    and not isinstance(variables, AbstractDstPort)):
            raise AssertionError(
                "Currently only 1:1 connections are supported for RefPorts:"
                " {!r}: {!r}".format(
                    self.process.__class__.__name__, self.name))

        variables: ty.List[Var] = to_list(variables)
        # Check all 'variables' are actually Vars and don't have same parent
        # process as RefPort
        for v in variables:
            if not isinstance(v, Var):
                raise AssertionError(
                    "'variables' must be a Var or list of Vars but "
                    "found {}.".format(v.__class__)
                )
            if self.process is not None:
                # Only assign when parent process is already assigned
                if self.process == v.process:
                    raise AssertionError("RefPort and Var have same "
                                         "parent process.")
        var_ports = []
        var_shape = variables[0].shape
        for v in variables:
            # Check that shapes of all vars are the same
            if var_shape != v.shape:
                raise AssertionError("All 'vars' must have same shape.")
            # Create a VarPort to wrap Var
            vp = self.create_implicit_var_port(v)
            var_ports.append(vp)
        # Connect RefPort to VarPorts that wrap Vars
        self.connect(var_ports, to_list(connection_configs))

    def get_dst_vars(self) -> ty.List[Var]:
        """Returns destination Vars this RefPort is connected to."""
        return [ty.cast(VarPort, p).var for p in self.get_dst_ports()]

    @staticmethod
    def create_implicit_var_port(var: Var) -> ImplicitVarPort:
        """Creates and returns an ImplicitVarPort for the given Var."""
        # Create a VarPort to wrap Var
        vp = ImplicitVarPort(var)
        # Propagate name and parent process of Var to VarPort
        vp.name = "_" + var.name + "_implicit_port"
        if var.process is not None:
            # Only assign when parent process is already assigned
            vp.process = var.process
            # VarPort name could shadow existing attribute
            if hasattr(var.process, vp.name):
                name = str(vp.name)
                name_suffix = 1
                while hasattr(var.process, vp.name):
                    vp.name = name + "_" + str(name_suffix)
                    name_suffix += 1
            setattr(var.process, vp.name, vp)
            var.process.var_ports.add_members({vp.name: vp})

        return vp


class VarPort(AbstractRVPort, AbstractDstPort):
    """VarPorts are members of a Lava Process and act as a wrapper for
    internal Lava Vars to facilitate connections between RefPorts and Vars
    for shared memory access from the parent process of the RefPort to
    the parent process of the Var.

    Shared-memory-based communication can have side-effects and should
    therefore be used with caution.

    VarPorts can receive connections from other RefPorts of peer processes
    or from other VarPorts of processes that contain this VarPort's parent
    process as a sub process via the connect(..) method. Similarly, VarPorts
    can connect to other VarPorts of nested sub processes via the
    connect_from(..) method.

    VarPorts can either be created in the constructor of a Process to
    explicitly expose a Var for shared memory access (which might be safer).
    Alternatively, VarPorts can be created dynamically by connecting a
    RefPort to a Var via the RefPort.connect_var(..) method."""

    def __init__(self, var: Var):
        if not isinstance(var, Var):
            raise AssertionError("'var' must be of type Var.")
        if not var.shareable:
            raise pe.VarNotSharableError(var.name)
        AbstractRVPort.__init__(self, var.shape)
        self.var = var

    def connect(self, ports: ty.Union["VarPort", ty.List["VarPort"]],
                connection_configs: ty.Optional[ConnectionConfigs] = None):
        """Connects this VarPort to other VarPort(s) of a nested process.
        VarPorts cannot connect to other RefPorts.

        Parameters
        ----------
        ports: ty.Union["VarPort", ty.List["VarPort"]]
            The VarPort(s) to connect to.
        connection_configs: ConnectionConfigs
            Configuration for this connection. See "ConnectionConfig" class.
        """

        # Check if multiple ports should be connected (currently not supported)
        if len(to_list(ports)) > 1 \
                or (len(self.get_dst_ports()) > 0
                    and not isinstance(ports, AbstractSrcPort)) \
                or (len(self.get_src_ports()) > 0
                    and not isinstance(ports, AbstractDstPort)):
            raise AssertionError(
                "Currently only 1:1 connections are supported for VarPorts:"
                " {!r}: {!r}".format(
                    self.process.__class__.__name__, self.name))

        for p in to_list(ports):
            if not isinstance(p, VarPort):
                raise TypeError(
                    "VarPorts can only be connected to VarPorts: "
                    "{!r}: {!r} -> {!r}: {!r}".format(
                        self.process.__class__.__name__, self.name,
                        p.process.__class__.__name__, p.name))
        self._connect_forward(to_list(ports),
                              VarPort,
                              to_list(connection_configs))

    def connect_from(
            self, ports: ty.Union["AbstractRVPort", ty.List["AbstractRVPort"]],
            connection_configs: ty.Optional[ConnectionConfigs] = None):
        """Connects other RefPort(s) to this VarPort or connects other
        VarPort(s) of parent process to this VarPort.

        Parameters
        ----------
        ports: ty.Union["AbstractRVPort", ty.List["AbstractRVPort"]]
            The AbstractRVPort(s) that connect to this VarPort.
        connection_configs: ConnectionConfigs
            Configuration for this connection. See "ConnectionConfig" class.
        """

        # Check if multiple ports should be connected (currently not supported)
        if len(to_list(ports)) > 1 \
                or (len(self.get_dst_ports()) > 0
                    and not isinstance(ports, AbstractSrcPort)) \
                or (len(self.get_src_ports()) > 0
                    and not isinstance(ports, AbstractDstPort)):
            raise AssertionError(
                "Currently only 1:1 connections are supported for VarPorts:"
                " {!r}: {!r}".format(
                    self.process.__class__.__name__, self.name))

        for p in to_list(ports):
            if not isinstance(p, RefPort) and not isinstance(p, VarPort):
                raise TypeError(
                    "VarPorts can only receive connections from RefPorts or "
                    "VarPorts: {!r}: {!r} -> {!r}: {!r}".format(
                        self.process.__class__.__name__, self.name,
                        p.process.__class__.__name__, p.name))
        self._connect_backward(to_list(ports),
                               AbstractRVPort,
                               to_list(connection_configs))


class ImplicitVarPort(VarPort):
    """Sub class for VarPort to identify implicitly created VarPorts when
    a RefPort connects directly to a Var."""

    def __init__(self, var: Var) -> None:
        super().__init__(var)


class AbstractVirtualPort(AbstractPort):
    """Abstract base class interface for any type of port that merely serves
    to transform the properties of a user-defined port."""

    @property
    def _parent_port(self):
        """Must return parent port that this VirtualPort was derived from."""
        return self.get_src_ports()[0]

    @property
    def process(self):
        """Returns parent process of parent port that this VirtualPort was
        derived from."""
        return self._parent_port.process

    def connect(self,
                ports: ty.Union["AbstractPort", ty.List["AbstractPort"]],
                connection_configs: ty.Optional[ConnectionConfigs] = None):
        """Connects this virtual port to other port(s).

        Parameters
        ----------
        ports: ty.Union["AbstractPort", ty.List["AbstractPort"]]
            The port(s) to connect to. Connections from an IOPort to a RVPort
            and vice versa are not allowed.
        connection_configs: ConnectionConfigs
            Configuration for this connection. See "ConnectionConfig" class.
        """
        # Determine allows port_type
        if isinstance(self._parent_port, OutPort):
            # If OutPort, only allow other IO ports
            port_type = AbstractIOPort
        elif isinstance(self._parent_port, InPort):
            # If InPort, only allow other InPorts
            port_type = InPort
        elif isinstance(self._parent_port, RefPort):
            # If RefPort, only allow other Ref- or VarPorts
            port_type = AbstractRVPort
        elif isinstance(self._parent_port, VarPort):
            # If VarPort, only allow other VarPorts
            port_type = VarPort
        else:
            raise TypeError("Illegal parent port.")
        # Connect to ports
        self._connect_forward(to_list(ports),
                              port_type,
                              to_list(connection_configs))

    @abstractmethod
    def get_transform_func_fwd(self) -> ft.partial:
        """Returns a function pointer that implements the forward (fwd)
        transformation of the virtual port.

        Returns
        -------
        function_pointer : functools.partial
            a function pointer that can be applied to incoming data"""
        pass

    @abstractmethod
    def get_transform_func_bwd(self) -> ft.partial:
        """Returns a function pointer that implements the backward (bwd)
        transformation of the virtual port.

        Returns
        -------
        function_pointer : functools.partial
            a function pointer that can be applied to incoming data"""
        pass


class ReshapePort(AbstractVirtualPort):
    """A ReshapePort is a virtual port that allows to change the shape of a
    port before connecting to another port.
    It is used by the compiler to map the indices of the underlying
    tensor-valued data array from the derived to the new shape."""

    def __init__(self,
                 new_shape: ty.Tuple[int, ...],
                 old_shape: ty.Tuple[int, ...]):
        super().__init__(new_shape)
        self.old_shape = old_shape

    def get_transform_func_fwd(self) -> ft.partial:
        """Returns a function pointer that implements the forward (fwd)
        transformation of the ReshapePort, which reshapes incoming data to
        a new shape (the shape of the destination Process).

        Returns
        -------
        function_pointer : functools.partial
            a function pointer that can be applied to incoming data"""
        return ft.partial(np.reshape, newshape=self.shape)

    def get_transform_func_bwd(self) -> ft.partial:
        """Returns a function pointer that implements the backward (bwd)
        transformation of the ReshapePort, which reshapes incoming data to
        a new shape (the shape of the source Process).

        Returns
        -------
        function_pointer : functools.partial
            a function pointer that can be applied to incoming data"""
        return ft.partial(np.reshape, newshape=self.old_shape)


class ConcatPort(AbstractVirtualPort):
    """A ConcatPort is a virtual port that allows to concatenate multiple
    ports along given axis into a new port before connecting to another port.
    The shape of all concatenated ports outside of the concatenation
    dimension must be the same.
    It is used by the compiler to map the indices of the underlying
    tensor-valued data array from the derived to the new shape."""

    def __init__(self, ports: ty.List[AbstractPort], axis: int):
        super().__init__(self._get_new_shape(ports, axis))
        self._connect_backward(
            ports, AbstractPort, [None], assert_same_shape=False,
            assert_same_type=True
        )
        self.concat_axis = axis

    @staticmethod
    def _get_new_shape(ports: ty.List[AbstractPort], axis):
        """Computes shape of ConcatPort from given 'ports'."""
        # Extract shapes of given ports
        concat_shapes = [p.shape for p in ports]
        total_size = 0
        shapes_ex_axis = []
        shapes_incompatible = False
        for shape in concat_shapes:
            if axis >= len(shape):
                raise pe.ConcatIndexError(shape, axis)

            # Compute total size along concatenation axis
            total_size += shape[axis]
            # Extract shape dimensions other than concatenation axis
            shapes_ex_axis.append(shape[:axis] + shape[axis + 1:])
            if len(shapes_ex_axis) > 1:
                shapes_incompatible = shapes_ex_axis[-2] != shapes_ex_axis[-1]

        if shapes_incompatible:
            raise pe.ConcatShapeError(shapes_ex_axis, axis)

        # Return shape of concatenated port
        new_shape = shapes_ex_axis[0]
        return new_shape[:axis] + (total_size,) + new_shape[axis:]

    def get_transform_func_fwd(self) -> ft.partial:
        raise NotImplementedError()

    def get_transform_func_bwd(self) -> ft.partial:
        raise NotImplementedError()


class TransposePort(AbstractVirtualPort):
    """A TransposePort is a virtual port that allows to permute the dimensions
    of a port before connecting to another port.
    It is used by the compiler to map the indices of the underlying
    tensor-valued data array from the derived to the new shape.

    Example:
        out_port = OutPort((2, 4, 3))
        in_port = InPort((3, 2, 4))
        out_port.transpose([3, 1, 2]).connect(in_port)
    """

    def __init__(self,
                 new_shape: ty.Tuple[int, ...],
                 axes: ty.Tuple[int, ...]):
        self.axes = axes
        super().__init__(new_shape)

    def get_transform_func_fwd(self) -> ft.partial:
        """Returns a function pointer that implements the forward (fwd)
        transformation of the TransposePort, which transposes (permutes)
        incoming data according to a specific order of axes (to match the
        destination Process).

        Returns
        -------
        function_pointer : functools.partial
            a function pointer that can be applied to incoming data"""
        return ft.partial(np.transpose, axes=self.axes)

    def get_transform_func_bwd(self) -> ft.partial:
        """Returns a function pointer that implements the backward (bwd)
        transformation of the TransposePort, which transposes (permutes)
        incoming data according to a specific order of axes (to match the
        source Process).

        Returns
        -------
        function_pointer : functools.partial
            a function pointer that can be applied to incoming data"""
        return ft.partial(np.transpose, axes=np.argsort(self.axes))
