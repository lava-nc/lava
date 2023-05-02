# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty

import numpy as np
from scipy.sparse import csr_matrix
from lava.magma.compiler.builders.interfaces import AbstractProcessBuilder

from lava.magma.compiler.channels.interfaces import AbstractCspPort
from lava.magma.compiler.channels.pypychannel import CspRecvPort, CspSendPort
from lava.magma.compiler.utils import (
    PortInitializer,
    VarInitializer,
    VarPortInitializer,
)
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.ports import (
    AbstractPyIOPort,
    IdentityTransformer,
    PyInPort,
    PyOutPort,
    PyRefPort,
    PyVarPort,
    VirtualPortTransformer,
)
from lava.magma.core.model.py.type import LavaPyType


class PyProcessBuilder(AbstractProcessBuilder):
    """A PyProcessBuilder instantiates and initializes a PyProcessModel.

    The compiler creates a PyProcessBuilder for each PyProcessModel. In turn,
    the runtime, loads a PyProcessBuilder onto a compute node where it builds
    the PyProcessModel and its associated ports.

    In order to build the PyProcessModel, the builder inspects all LavaType
    class variables of a PyProcessModel, creates the corresponding data type
    with the specified properties, the shape and the initial value provided by
    the Lava Var. In addition, the builder creates the required PyPort
    instances. Finally, the builder assigns both port and variable
    implementations to the PyProcModel.

    Once the PyProcessModel is built, it is the RuntimeService's job to
    connect channels to ports and start the process.

    Note: For unit testing it should be possible to build processes locally
    instead of on a remote node. For pure atomic unit testing a ProcessModel
    locally, PyInPorts and PyOutPorts must be fed manually with data.


    """

    def __init__(
        self,
        proc_model: ty.Type[AbstractPyProcessModel],
        model_id: int,
        proc_params: ty.Dict[str, ty.Any] = None,
    ):
        super().__init__(proc_model=proc_model, model_id=model_id)
        if not issubclass(proc_model, AbstractPyProcessModel):
            raise AssertionError("Is not a subclass of AbstractPyProcessModel")
        self.vars: ty.Dict[str, VarInitializer] = {}
        self.py_ports: ty.Dict[str, PortInitializer] = {}
        self.ref_ports: ty.Dict[str, PortInitializer] = {}
        self.var_ports: ty.Dict[str, VarPortInitializer] = {}
        self.csp_ports: ty.Dict[str, ty.List[AbstractCspPort]] = {}
        self._csp_port_map: ty.Dict[str, ty.Dict[str, AbstractCspPort]] = {}
        self.csp_rs_send_port: ty.Dict[str, CspSendPort] = {}
        self.csp_rs_recv_port: ty.Dict[str, CspRecvPort] = {}
        self.proc_params = proc_params

    def check_all_vars_and_ports_set(self):
        """Checks that Vars and PyPorts assigned from Process have a
        corresponding LavaPyType.

        Raises
        ------
        AssertionError
            No LavaPyType found in ProcModel
        """
        for attr_name in dir(self.proc_model):
            attr = getattr(self.proc_model, attr_name)
            if isinstance(attr, LavaPyType):
                if (
                    attr_name not in self.vars
                    and attr_name not in self.py_ports
                    and attr_name not in self.ref_ports
                    and attr_name not in self.var_ports
                ):
                    raise AssertionError(
                        f"No LavaPyType '{attr_name}' found in ProcModel "
                        f"'{self.proc_model.__name__}'."
                    )

    def check_lava_py_types(self):
        """Checks correctness of LavaPyTypes.

        Any Py{In/Out/Ref}Ports must be strict sub-types of Py{In/Out/Ref}Ports.
        """
        for name, port_init in self.py_ports.items():
            lt = self._get_lava_type(name)
            if not isinstance(lt.cls, type):
                raise AssertionError(
                    f"LavaPyType.cls for '{name}' is not a type in '"
                    f"{self.proc_model.__name__}'."
                )
            if port_init.port_type == "InPort":
                if not (issubclass(lt.cls, PyInPort) and lt.cls != PyInPort):
                    raise AssertionError(
                        f"LavaPyType for '{name}' must be a strict sub-type "
                        f"of PyInPort in '{self.proc_model.__name__}'."
                    )
            elif port_init.port_type == "OutPort":
                if not (issubclass(lt.cls, PyOutPort) and lt.cls != PyOutPort):
                    raise AssertionError(
                        f"LavaPyType for '{name}' must be a strict sub-type of "
                        f"PyOutPort in '{self.proc_model.__name__}'."
                    )
            elif port_init.port_type == "RefPort":
                if not (issubclass(lt.cls, PyRefPort) and lt.cls != PyRefPort):
                    raise AssertionError(
                        f"LavaPyType for '{name}' must be a strict sub-type of "
                        f"PyRefPort in '{self.proc_model.__name__}'."
                    )

    def set_py_ports(self, py_ports: ty.List[PortInitializer], check=True):
        """Appends the given list of PyPorts to the ProcessModel. Used by the
         compiler to create a ProcessBuilder during the compilation of
         ProcessModels.

        Parameters
        ----------
        py_ports : ty.List[PortInitializer]

        check : bool, optional
             , by default True
        """
        if check:
            self._check_members_exist(py_ports, "Port")
        new_ports = {p.name: p for p in py_ports}
        self._check_not_assigned_yet(self.py_ports, new_ports.keys(), "ports")
        self.py_ports.update(new_ports)

    def set_ref_ports(self, ref_ports: ty.List[PortInitializer]):
        """Appends the given list of RefPorts to the ProcessModel. Used by the
         compiler to create a ProcessBuilder during the compilation of
         ProcessModels.

        Parameters
        ----------
        ref_ports : ty.List[PortInitializer]
        """
        self._check_members_exist(ref_ports, "Port")
        new_ports = {p.name: p for p in ref_ports}
        self._check_not_assigned_yet(self.ref_ports, new_ports.keys(), "ports")
        self.ref_ports.update(new_ports)

    def set_var_ports(self, var_ports: ty.List[VarPortInitializer]):
        """Appends the given list of VarPorts to the ProcessModel. Used by the
         compiler to create a ProcessBuilder during the compilation of
         ProcessModels.

        Parameters
        ----------
        var_ports : ty.List[VarPortInitializer]
        """
        new_ports = {p.name: p for p in var_ports}
        self._check_not_assigned_yet(self.var_ports, new_ports.keys(), "ports")
        self.var_ports.update(new_ports)

    def set_csp_ports(self, csp_ports: ty.List[AbstractCspPort]):
        """Appends the given list of CspPorts to the ProcessModel. Used by the
        runtime to configure csp ports during initialization (_build_channels).

        Parameters
        ----------
        csp_ports : ty.List[AbstractCspPort]


        Raises
        ------
        AssertionError
            PyProcessModel has no port of that name
        """
        new_ports = {}
        for p in csp_ports:
            new_ports.setdefault(p.name, []).extend(
                p if isinstance(p, list) else [p]
            )

        # Check that there's a PyPort for each new CspPort
        proc_name = self.proc_model.implements_process.__name__
        for port_name in new_ports:
            if not hasattr(self.proc_model, port_name):
                raise AssertionError(
                    "PyProcessModel '{}' has \
                    no port named '{}'.".format(
                        proc_name, port_name
                    )
                )

            if port_name in self.csp_ports:
                self.csp_ports[port_name].extend(new_ports[port_name])
            else:
                self.csp_ports[port_name] = new_ports[port_name]

    def add_csp_port_mapping(self, py_port_id: str, csp_port: AbstractCspPort):
        """Appends a mapping from a PyPort ID to a CSP port. This is used
        to associate a CSP port in a PyPort with transformation functions
        that implement the behavior of virtual ports.

        Parameters
        ----------
        py_port_id : str
            ID of the PyPort that contains the CSP on the other side of the
            channel of 'csp_port'
        csp_port : AbstractCspPort
            a CSP port
        """
        # Add or update the mapping
        self._csp_port_map.setdefault(csp_port.name, {}).update(
            {py_port_id: csp_port}
        )

    def set_rs_csp_ports(self, csp_ports: ty.List[AbstractCspPort]):
        """Set RS CSP Ports

        Parameters
        ----------
        csp_ports : ty.List[AbstractCspPort]

        """
        for port in csp_ports:
            if isinstance(port, CspSendPort):
                self.csp_rs_send_port.update({port.name: port})
            if isinstance(port, CspRecvPort):
                self.csp_rs_recv_port.update({port.name: port})

    def _get_lava_type(self, name: str) -> LavaPyType:
        return getattr(self.proc_model, name)

    def build(self):
        """Builds a PyProcModel at runtime within Runtime.

        The Compiler initializes the PyProcBuilder with the ProcModel,
        VarInitializers and PortInitializers.
        The Runtime builds the channels and CSP ports between all ports,
        assigns them to builder.

        At deployment to a node, the Builder.build(..) gets executed
        resulting in the following:

          1. ProcModel gets instantiated
          2. Vars are initialized and assigned to ProcModel
          3. PyPorts are initialized (with CSP ports) and assigned to ProcModel

        Returns
        -------
        AbstractPyProcessModel


        Raises
        ------
        NotImplementedError
        """

        # Create the ProcessModel
        # Default value of pm.proc_params in ProcessModel is an empty dictionary
        # If a proc_params argument is provided in PyProcessBuilder,
        # this will be carried to ProcessModel
        pm = self.proc_model(self.proc_params)
        pm.model_id = self._model_id

        # Initialize PyPorts
        for name, p in self.py_ports.items():
            # Build PyPort
            lt = self._get_lava_type(name)
            port_cls = ty.cast(ty.Type[AbstractPyIOPort], lt.cls)
            csp_ports = []
            if name in self.csp_ports:
                csp_ports = self.csp_ports[name]
                if not isinstance(csp_ports, list):
                    csp_ports = [csp_ports]

            if issubclass(port_cls, PyInPort):
                transformer = (
                    VirtualPortTransformer(
                        self._csp_port_map[name], p.transform_funcs
                    )
                    if p.transform_funcs
                    else IdentityTransformer()
                )
                port_cls = ty.cast(ty.Type[PyInPort], lt.cls)
                port = port_cls(csp_ports, pm, p.shape, lt.d_type, transformer)
            elif issubclass(port_cls, PyOutPort):
                port = port_cls(csp_ports, pm, p.shape, lt.d_type)
            else:
                raise AssertionError(
                    "port_cls must be of type PyInPort or " "PyOutPort"
                )

            # Create dynamic PyPort attribute on ProcModel
            setattr(pm, name, port)
            # Create private attribute for port precision
            # setattr(pm, "_" + name + "_p", lt.precision)

        # Initialize RefPorts
        for name, p in self.ref_ports.items():
            # Build RefPort
            lt = self._get_lava_type(name)
            port_cls = ty.cast(ty.Type[PyRefPort], lt.cls)
            csp_recv = None
            csp_send = None
            if name in self.csp_ports:
                csp_ports = self.csp_ports[name]
                csp_recv = (
                    csp_ports[0]
                    if isinstance(csp_ports[0], CspRecvPort)
                    else csp_ports[1]
                )
                csp_send = (
                    csp_ports[0]
                    if isinstance(csp_ports[0], CspSendPort)
                    else csp_ports[1]
                )

            transformer = (
                VirtualPortTransformer(
                    self._csp_port_map[name], p.transform_funcs
                )
                if p.transform_funcs
                else IdentityTransformer()
            )

            port = port_cls(
                csp_send, csp_recv, pm, p.shape, lt.d_type, transformer
            )

            # Create dynamic RefPort attribute on ProcModel
            setattr(pm, name, port)

        # Initialize VarPorts
        for name, p in self.var_ports.items():
            # Build VarPort
            if p.port_cls is None:
                # VarPort is not connected
                continue
            port_cls = ty.cast(ty.Type[PyVarPort], p.port_cls)
            csp_recv = None
            csp_send = None
            if name in self.csp_ports:
                csp_ports = self.csp_ports[name]
                csp_recv = (
                    csp_ports[0]
                    if isinstance(csp_ports[0], CspRecvPort)
                    else csp_ports[1]
                )
                csp_send = (
                    csp_ports[0]
                    if isinstance(csp_ports[0], CspSendPort)
                    else csp_ports[1]
                )

            transformer = (
                VirtualPortTransformer(
                    self._csp_port_map[name], p.transform_funcs
                )
                if p.transform_funcs
                else IdentityTransformer()
            )

            port = port_cls(
                p.var_name,
                csp_send,
                csp_recv,
                pm,
                p.shape,
                p.d_type,
                transformer,
            )

            # Create dynamic VarPort attribute on ProcModel
            setattr(pm, name, port)

        for port in self.csp_rs_recv_port.values():
            if "service_to_process" in port.name:
                pm.service_to_process = port
                continue

        for port in self.csp_rs_send_port.values():
            if "process_to_service" in port.name:
                pm.process_to_service = port
                continue

        # Initialize Vars
        for name, v in self.vars.items():
            # Build variable
            lt = self._get_lava_type(name)
            if issubclass(lt.cls, np.ndarray):
                var = lt.cls(v.shape, lt.d_type)
                var[:] = v.value
            elif issubclass(lt.cls, (int, float, str)):
                var = v.value
            elif issubclass(lt.cls, (csr_matrix)):
                if isinstance(v.value, int):
                    var = csr_matrix(v.shape, dtype=lt.d_type)
                    var[:] = v.value
                else:
                    var = v.value
            else:
                raise NotImplementedError(
                    "Cannot initiliaze variable "
                    "datatype, \
                    only subclasses of int, float and str are \
                    supported"
                )

            # Create dynamic variable attribute on ProcModel
            setattr(pm, name, var)
            # Create private attribute for variable precision
            setattr(pm, "_" + name + "_p", lt.precision)

            pm.var_id_to_var_map[v.var_id] = name

        return pm
