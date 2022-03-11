# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import logging
import importlib
import importlib.util as import_utils
import inspect
import os
import pkgutil
import sys
import types
import typing as ty
from collections import OrderedDict, defaultdict
from warnings import warn

import numpy as np

import lava.magma.compiler.exceptions as ex
import lava.magma.compiler.exec_var as exec_var
from lava.magma.compiler.builders.builder import ChannelBuilderMp
from lava.magma.compiler.builders.builder import PyProcessBuilder, \
    NcProcessBuilder, AbstractRuntimeServiceBuilder, RuntimeServiceBuilder, \
    AbstractChannelBuilder, ServiceChannelBuilderMp
from lava.magma.compiler.builders.builder import RuntimeChannelBuilderMp
from lava.magma.compiler.channels.interfaces import ChannelType
from lava.magma.compiler.executable import Executable
from lava.magma.compiler.node import NodeConfig, Node
from lava.magma.compiler.utils import VarInitializer, PortInitializer, \
    VarPortInitializer
from lava.magma.core import resources
from lava.magma.core.model.c.model import AbstractCProcessModel
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.nc.model import (
    AbstractNcProcessModel,
    NcProcessModel
)
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.ports import RefVarTypeMapping
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import AbstractPort, VarPort, \
    ImplicitVarPort, InPort, RefPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import (
    CPU,
    Loihi1NeuroCore,
    Loihi2NeuroCore,
    NeuroCore
)
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.runtime.runtime import Runtime
from lava.magma.runtime.runtime_services.enums import LoihiVersion

PROC_MAP = ty.Dict[AbstractProcess, ty.Type[AbstractProcessModel]]


# ToDo: (AW) Document all class methods and class
class Compiler:
    def __init__(self, loglevel: int = logging.WARNING,
                 compile_cfg: ty.Optional[ty.Dict[str, ty.Any]] = None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(loglevel)
        self._compile_config = {"pypy_channel_size": 64}
        if compile_cfg:
            self._compile_config.update(compile_cfg)

    # ToDo: (AW) Clean this up by avoiding redundant search paths
    def _find_processes(self,
                        proc: AbstractProcess,
                        seen_procs: ty.List[AbstractProcess] = None) \
            -> ty.List[AbstractProcess]:
        """Finds and returns  list of all processes at same level of
        hierarchy that are connected to 'proc'.

        Processes are connected via different kinds of Ports to other
        processes. This method starts at the given 'proc' and traverses the
        graph along connections to find all other connected processes."""

        # processes which have been processed already (avoid loops)
        seen_procs = [] if seen_procs is None else seen_procs

        # add the process compile was called on (main process)
        seen_procs.append(proc)
        new_list: ty.List[AbstractProcess] = []

        # add processes connecting to the main process
        for in_port in proc.in_ports.members + proc.var_ports.members:
            for con in in_port.get_src_ports():
                new_list.append(con.process)
            for con in in_port.get_dst_ports():
                new_list.append(con.process)

        # add processes connecting from the main process
        for out_port in proc.out_ports.members + proc.ref_ports.members:
            for con in out_port.get_src_ports():
                new_list.append(con.process)
            for con in out_port.get_dst_ports():
                new_list.append(con.process)

        for proc in set(new_list):
            if proc not in seen_procs:
                new_list.extend(self._find_processes(proc, seen_procs))

        seen_procs.extend(new_list)
        seen_procs = list(set(seen_procs))

        return seen_procs

    @staticmethod
    def _find_proc_models_in_module(proc: AbstractProcess,
                                    module: types.ModuleType) \
            -> ty.List[ty.Type[AbstractProcessModel]]:
        """Finds and returns all ProcModels that implement given Process in
        given Python module."""

        proc_models = []
        for name, cls in module.__dict__.items():
            if (
                    hasattr(cls, "implements_process")
                    and issubclass(cls, AbstractProcessModel)
                    and cls.implements_process is not None
                    and isinstance(proc, cls.implements_process)
            ):
                # Collect ProcessModel
                proc_models.append(cls)

        return proc_models

    # ToDo: (AW) This is a mess and should be cleaned up
    def _find_proc_models(self, proc: AbstractProcess) \
            -> ty.List[ty.Type[AbstractProcessModel]]:
        """Finds all ProcessModels that implement given Process.

        First, we search in the same Python module, in which 'proc' is defined,
        for ProcModels implementing 'proc'. Next, we search through modules in
        the same directory.
        """

        # Find all ProcModel classes that implement 'proc' in same module
        proc_module = sys.modules[proc.__module__]
        proc_models = self._find_proc_models_in_module(proc, proc_module)

        # Find all ProcModel classes that implement 'proc' in same directory
        # search for the file of the module
        # ToDo: (AW) I think this could be simplified by using checking if
        #  get_ipython exists and then use it to get get_ipython(
        #  ).user_global_ns. This will return globally known classes,
        #  including those in the jupyter namespace and therefore allow
        #  using any ProcModels defined there.
        file = None
        if inspect.isclass(proc.__class__):
            if hasattr(proc.__class__, '__module__'):
                proc_ = sys.modules.get(proc.__class__.__module__)
                # check if it has file (classes in jupyter nb do not)
                if hasattr(proc_, '__file__'):
                    file = proc_.__file__
            else:
                raise TypeError('Source for {!r} not found'.format(object))

            if file is None:
                # class is probably defined in a jupyter notebook
                # lookup file name per methods
                for _, m in inspect.getmembers(proc.__class__):
                    if inspect.isfunction(m) and \
                            proc.__class__.__qualname__ + '.' \
                            + m.__name__ == m.__qualname__:
                        file = inspect.getfile(m)
                        break
        else:
            file = inspect.getfile(proc.__class__)

        dir_name = os.path.dirname(file)
        for _, name, _ in pkgutil.iter_modules([dir_name]):
            import_path = os.path.join(dir_name, name)
            name_not_dir = not os.path.isdir(import_path)
            if name_not_dir:
                spec = import_utils.spec_from_file_location(
                    name, os.path.join(dir_name, name + ".py"))
                module = import_utils.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    if module != proc_module:
                        pm = self._find_proc_models_in_module(proc, module)
                        # TODO: Remove this hack of getting the qualified
                        #  name path of the class in this manner
                        for proc_model in pm:
                            proc_cls_mod = \
                                inspect.getmodule(proc).__package__ + '.' + \
                                proc_model.__module__
                            proc_cls_mod = importlib. \
                                import_module(proc_cls_mod)
                            class_ = getattr(proc_cls_mod, proc_model.__name__)
                            proc_models.append(class_)
                except Exception:
                    warn(f"Cannot import module '{module}' when searching "
                         f"ProcessModels for Process "
                         f"'{proc.__class__.__name__}'.")

        if not proc_models:
            raise ex.NoProcessModelFound(proc)
        return proc_models

    @staticmethod
    def _select_proc_models(
            proc: AbstractProcess,
            models: ty.List[ty.Type[AbstractProcessModel]],
            run_cfg: RunConfig) -> ty.Type[AbstractProcessModel]:
        """Selects a ProcessModel from list of provided models given RunCfg."""
        selected_proc_model = run_cfg.select(proc, models)

        if not isinstance(selected_proc_model, type) \
                or not issubclass(selected_proc_model, AbstractProcessModel):
            err_msg = f"RunConfig {run_cfg.__class__.__qualname__}.select()" \
                      f" must return a sub-class of AbstractProcessModel. Got" \
                      f" {type(selected_proc_model)} instead."
            raise AssertionError(err_msg)

        return selected_proc_model

    @staticmethod
    def _propagate_var_ports(proc: AbstractProcess):
        """Checks if the process has VarPorts configured and if they reference
        an aliased Var. If this is the case an implicit VarPort is created in
        the sub process and the VarPort is connected to the new implicit
        VarPort.
        This is needed since a RefPort or VarPort of a sub process cannot be
         directly targeted from processes of a different parent process."""

        for vp in proc.var_ports:
            v = vp.var.aliased_var
            if v is not None:
                # Create an implicit Var port in the sub process
                imp_vp = RefPort.create_implicit_var_port(v)
                # Connect the VarPort to the new VarPort
                vp.connect(imp_vp)

    def _expand_sub_proc_model(self,
                               model_cls: ty.Type[AbstractSubProcessModel],
                               proc: AbstractProcess,
                               run_cfg: RunConfig) -> PROC_MAP:
        """Expands a SubProcModel by building SubProcModel, extracting sub
        processes and mapping sub processes recursively to its process
        models."""

        # Build SubProcessModel
        model = model_cls(proc)
        # Check if VarPorts are configured and propagate them to the sub process
        self._propagate_var_ports(proc)
        # Discover sub processes and register with parent process
        sub_procs = model.find_sub_procs()
        proc.register_sub_procs(sub_procs)
        proc.validate_var_aliases()
        # Recursively map sub processes to their ProcModel
        return self._map_proc_to_model(list(sub_procs.values()), run_cfg)

    def _map_proc_to_model(self,
                           procs: ty.List[AbstractProcess],
                           run_cfg: RunConfig) -> PROC_MAP:
        """Maps each Process in 'procs' to Type[AbstractProcessModel] as
        selected by RunCfg. If a hierarchical process selected, it is expanded
        recursively."""

        proc_map = OrderedDict()
        for p in procs:
            # Select a specific ProcessModel
            models_cls = self._find_proc_models(p)
            model_cls = self._select_proc_models(p, models_cls, run_cfg)
            if issubclass(model_cls, AbstractSubProcessModel):
                # Recursively substitute SubProcModel by sub processes
                sub_map = self._expand_sub_proc_model(model_cls, p, run_cfg)
                proc_map.update(sub_map)
            else:
                # Just map current Process to selected ProcessModel
                proc_map[p] = model_cls

        return proc_map

    @staticmethod
    def _group_proc_by_model(proc_map: PROC_MAP) \
            -> ty.Dict[ty.Type[AbstractProcessModel],
                       ty.List[AbstractProcess]]:
        """Groups processes by ProcessModel by building a reverse map
        Type[ProcessModel] -> List[Process], which maps a ProcessModel type to
        the list of all Processes that get implemented by it."""

        grouped_models = OrderedDict(defaultdict(list))

        for proc in proc_map:
            model = proc_map[proc]
            if model in grouped_models:
                grouped_models[model].append(proc)
            else:
                grouped_models[model] = [proc]

        return grouped_models

    # TODO: (PP) This currently only works for PyPorts - needs general solution
    # TODO: (PP) Currently does not support 1:many/many:1 connections
    @staticmethod
    def _map_var_port_class(port: VarPort,
                            proc_groups: ty.Dict[ty.Type[AbstractProcessModel],
                                                 ty.List[AbstractProcess]]):
        """Maps the port class of a given VarPort from its source RefPort. This
        is needed as implicitly created VarPorts created by connecting RefPorts
        directly to Vars, have no LavaType."""

        # Get the source RefPort of the VarPort
        rp = port.get_src_ports()
        if len(rp) > 0:
            rp = rp[0]
        else:
            # VarPort is not connect, hence there is no LavaType
            return None

        # Get the ProcessModel of the source RefPort
        r_pm = None
        for pm in proc_groups:
            if rp.process in proc_groups[pm]:
                r_pm = pm

        # Get the LavaType of the RefPort from its ProcessModel
        lt = getattr(r_pm, rp.name)

        # Return mapping of the RefPort class to VarPort class
        return RefVarTypeMapping.get(lt.cls)

    # TODO: (PP) possible shorten creation of PortInitializers
    def _compile_proc_models(
            self,
            proc_groups: ty.Dict[ty.Type[AbstractProcessModel],
                                 ty.List[AbstractProcess]]) -> Executable:
        """Compiles ProcessModels by generating a Builder for each Process
        given its ProcessModel. Returns all Builders as part of an
        Executable."""

        py_builders = {}
        c_builders = {}
        nc_builders = {}
        pp_ch_size = self._compile_config["pypy_channel_size"]
        for pm, procs in proc_groups.items():
            if issubclass(pm, AbstractPyProcessModel):
                for p in procs:
                    b = PyProcessBuilder(pm, p.id, p.proc_params)
                    # Create Var- and PortInitializers from lava.process Vars
                    # and Ports
                    v = [VarInitializer(v.name, v.shape, v.init, v.id)
                         for v in p.vars]

                    ports = []
                    for pt in (list(p.in_ports) + list(p.out_ports)):
                        # For all InPorts that receive input from
                        # virtual ports...
                        transform_funcs = None
                        if isinstance(pt, InPort):
                            # ... extract a function pointer to the
                            # transformation function of each virtual port.
                            transform_funcs = pt.get_incoming_transform_funcs()

                        pi = PortInitializer(pt.name,
                                             pt.shape,
                                             self._get_port_dtype(pt, pm),
                                             pt.__class__.__name__,
                                             pp_ch_size,
                                             transform_funcs)
                        ports.append(pi)

                    # Create RefPort (also use PortInitializers)
                    ref_ports = []
                    for pt in list(p.ref_ports):
                        transform_funcs = pt.get_outgoing_transform_funcs()

                        pi = PortInitializer(pt.name,
                                             pt.shape,
                                             self._get_port_dtype(pt, pm),
                                             pt.__class__.__name__,
                                             pp_ch_size,
                                             transform_funcs)
                        ref_ports.append(pi)

                    # Create VarPortInitializers (contain also the Var name)
                    var_ports = []
                    for pt in list(p.var_ports):
                        transform_funcs = pt.get_incoming_transform_funcs()

                        pi = VarPortInitializer(
                            pt.name,
                            pt.shape,
                            pt.var.name,
                            self._get_port_dtype(pt, pm),
                            pt.__class__.__name__,
                            pp_ch_size,
                            self._map_var_port_class(pt, proc_groups),
                            transform_funcs)
                        var_ports.append(pi)

                        # Set implicit VarPorts (created by connecting a RefPort
                        # directly to a Var) as attribute to ProcessModel
                        if isinstance(pt, ImplicitVarPort):
                            setattr(pm, pt.name, pt)

                    # Assigns initializers to builder
                    b.set_variables(v)
                    b.set_py_ports(ports)
                    b.set_ref_ports(ref_ports)
                    b.set_var_ports(var_ports)
                    b.check_all_vars_and_ports_set()
                    py_builders[p] = b
            elif issubclass(pm, AbstractCProcessModel):
                raise NotImplementedError
            elif issubclass(pm, AbstractNcProcessModel):
                for p in procs:
                    b = NcProcessBuilder(pm, p.id, p.proc_params)
                    # Create VarInitializers from lava.process Vars
                    v = [VarInitializer(v.name, v.shape, v.init, v.id)
                         for v in p.vars]

                    # Assigns initializers to builder
                    b.set_variables(v)
                    b.check_all_vars_set()
                    nc_builders[p] = b
            else:
                raise TypeError("Non-supported ProcessModel type {}"
                                .format(pm))

        # Initialize Executable with ProcBuilders
        exe = Executable()
        exe.set_py_builders(py_builders)
        exe.set_c_builders(c_builders)
        exe.set_nc_builders(nc_builders)

        return exe

    @staticmethod
    def _create_sync_domains(
            proc_map: PROC_MAP, run_cfg: RunConfig, node_cfgs,
            log: logging.getLoggerClass()
    ) -> ty.Tuple[ty.List[SyncDomain], ty.Dict[Node, ty.List[SyncDomain]]]:
        """Validates custom sync domains provided by run_cfg and otherwise
        creates default sync domains.

        Users can manually create custom sync domains with a certain sync
        protocol and assign processes to these sync domains. The process
        models chosen for these processes must implement that sync protocol.

        If processes are not manually assigned to sync domains, the compiler
        will manually create one default sync domain for each unique sync
        protocol found among all process models and assign the remaining
        unassigned processes to those default sync domains based on the sync
        protocol that the chosen process model implements.
        """
        proc_to_domain_map = OrderedDict()
        sync_domains = OrderedDict()

        # Validate custom sync domains and map processes to their sync domain
        for sd in run_cfg.custom_sync_domains:
            # Sync domain names are used as identifiers, therefore they must be
            # unique
            if sd.name in sync_domains:
                raise AssertionError(
                    f"SyncDomain names must be unique but found domain name '"
                    f"{sd.name}' at least twice."
                )

            # Validate and map all processes in sync domain
            for p in sd.processes:
                log.debug("Process: " + str(p))
                pm = proc_map[p]

                # Auto-assign AsyncProtocol if none was assigned
                if not pm.implements_protocol:
                    proto = AsyncProtocol
                    log.debug("Protocol: AsyncProtocol")
                else:
                    proto = pm.implements_protocol
                    log.debug("Protocol: " + proto.__name__)

                # Check that SyncProtocols of process model and sync domain
                # are compatible
                if not isinstance(sd.protocol, proto):
                    raise AssertionError(
                        f"ProcessModel '{pm.__name__}' of Process "
                        f"'{p.name}::{p.__class__.__name__}' implements "
                        f"SyncProtocol '{proto.__name__}' which "
                        f"is not compatible with SyncDomain '{sd.name}' using "
                        f"SyncProtocol '{sd.protocol.__class__.__name__}'."
                    )

                # Map process to sync domain
                if p not in proc_to_domain_map:
                    proc_to_domain_map[p] = sd
                else:
                    # Processes can only be in one sync domain
                    raise AssertionError(
                        f"Processes can only be assigned to one sync domain "
                        f"but Process '{p.name}' has been found in SyncDomains"
                        f" '{proc_to_domain_map[p].name}' and '{sd.name}'."
                    )

            # Collect legal sync domains
            sync_domains[sd.name] = sd

        # Map remaining processes to their sync domains
        for p, pm in proc_map.items():
            # Auto-assign AsyncProtocol if none was assigned
            if not pm.implements_protocol:
                proto = AsyncProtocol
                log.debug("Protocol: AsyncProtocol")
            else:
                proto = pm.implements_protocol
                log.debug("Protocol: " + proto.__name__)

            # Add process to existing or new default sync domain if not part
            # of custom sync domain
            if p not in proc_to_domain_map:
                default_sd_name = proto.__name__ + "_SyncDomain"
                if default_sd_name in sync_domains:
                    # Default sync domain for current protocol already exists
                    sd = sync_domains[default_sd_name]
                else:
                    # ... or does not exist yet
                    sd = SyncDomain(name=default_sd_name, protocol=proto())
                    sync_domains[default_sd_name] = sd
                sd.add_process(p)
                proc_to_domain_map[p] = sd

        node_to_sync_domain_dict: ty.Dict[Node, ty.List[SyncDomain]] = \
            defaultdict(list)
        for node_cfg in node_cfgs:
            for node in node_cfg:
                log.debug("Node: " + str(node.node_type.__name__))
                node_to_sync_domain_dict[node].extend(
                    [proc_to_domain_map[proc] for proc in node.processes])
        return list(sync_domains.values()), node_to_sync_domain_dict

    # ToDo: (AW) Implement the general NodeConfig generation algorithm
    @staticmethod
    def _create_node_cfgs(proc_map: PROC_MAP,
                          log: logging.getLoggerClass()
                          ) -> ty.List[NodeConfig]:
        """Creates and returns a list of NodeConfigs from the
        AbstractResource requirements of all process's ProcessModels where
        each NodeConfig is a set of Nodes that satisfies the resource
        requirements of all processes.

        A NodeCfg is a set of Nodes. A Node has a particular AbstractNode
        type and contains a list of processes assigned to this Node. All
        Nodes in a NodeConfig taken together must satisfy the
        AbstractResource requirements of all available processes.

        The goal of this function is to find first find all legal
        NodeConfigs but then to assign processes to the Nodes of each
        NodeConfig and find the best, most optimal or minimal set of such
        NodeConfigs.

        To start with, we just hard code to return a NodeConfig with a single
        Node that is of type 'HeadNode' because that's all we need for pure
        Python execution. Once we proceed to supporting Loihi systems,
        we will enable the full algorithm outlined below.

        Algo:
        -----
        Step 1: Find which node supports which process

        - Get list of all processes -> Assume N of them
        - Get list of all available AbstractNodes -> Assume M of them
        - Initialize (N x M) boolean array to associate which node satisfies
          which process's resource requirements and initialize with False

        - Populate this array by iterating over all processes. For each
        process:
            - Get AbstractResource requirements from lava.process model
            - Perhaps even get quantitative resource capacity requirements
              from compiled process (i.e. num cores from NcProcBuilder)
            - Find nodes that satisfy resource requirements
            - Set array[i, j] for process i and node j to True if node
            satisfies process's resource requirements

        Step 2: Find all sets of legal combinations of nodes that support
        resource requirements of all N processes.
        Node: Each such set is a particular column combination of the array;
        represented by a binary index vector into the columns of the array.
        Since there are 2**M possible combinations of columns, the most
        naive way to find all sets is just to iterate over all 2**M
        combinations. If we would expect large M, then this would be very
        bad. But this can be optimized later.

        - Initialize empty list of legal node combinations
        - for i in range(2**M):
              - Convert i into binary vector v
              - Use v to index into columns of array to get column vector of
                processes supported by each node
              - OR all those columns together
              - If the sum of the OR'ed vectors sums to N, then it means that
                this particular node combination was legal to support ll
                process's resource requirements. If so, add i to a list of
                legal node combinations
        -> In the end we have a list of integers, whose binary representation
        corresponds to the legal node configurations that supports all
        processes

        Step 3: Assign the processes to specific nodes within the sets of
        legal node configurations and pick the best or all of them.
        Note: There could be multiple ways of doing this. I.e. in a cluster
        of CPUs one might strive to equally balance them with processes. In
        case of Loihi systems with limited resources one must not exceed them
        (but that might have already been taken care of through determining
        which is legal). But for Loihi resources one might also want to avoid
        using too many different ones; i.e. one would not put them on 2
        different Nahukus if they fit on one, etc.

         Finally, we are left with a list of (the best) legal NodeCfgs.
        """
        procs = list(proc_map.keys())
        if log.level == logging.DEBUG:
            for proc in procs:
                log.debug("Proc Name: " + proc.name + " Proc: " + str(proc))
            proc_models = list(proc_map.items())
            for procm in proc_models:
                log.debug("ProcModels: " + str(procm[1]))

        n = Node(node_type=resources.HeadNode, processes=procs)
        ncfg = NodeConfig()
        ncfg.append(n)

        # Until NodeConfig generation algorithm is present
        # check if NcProcessModel is present in proc_map,
        # if so add hardcoded Node for OheoGulch
        for proc_model in proc_map.items():
            if issubclass(proc_model[1], NcProcessModel):
                n1 = Node(node_type=resources.OheoGulch, processes=procs)
                ncfg.append(n1)
                log.debug("OheoGulch Node Added to NodeConfig: "
                          + str(n1.node_type))

        return [ncfg]

    @staticmethod
    def _get_channel_type(src: ty.Type[AbstractProcessModel],
                          dst: ty.Type[AbstractProcessModel]) \
            -> ChannelType:
        """Returns appropriate ChannelType for a given (source, destination)
        pair of ProcessModels."""

        if issubclass(src, AbstractPyProcessModel) and issubclass(
                dst, AbstractPyProcessModel
        ):
            return ChannelType.PyPy
        else:
            raise NotImplementedError(
                f"No support for (source, destination) pairs of type "
                f"'({src.__name__}, {dst.__name__})' yet."
            )

    @staticmethod
    def _get_port_dtype(port: AbstractPort,
                        proc_model: ty.Type[AbstractProcessModel]) -> type:
        """Returns the d_type of a Process Port, as specified in the
        corresponding PortImplementation of the ProcessModel implementing the
        Process"""

        # In-, Out-, Ref- and explicit VarPorts
        if hasattr(proc_model, port.name):
            # Handle VarPorts (use dtype of corresponding Var)
            if isinstance(port, VarPort):
                return getattr(proc_model, port.var.name).d_type
            return getattr(proc_model, port.name).d_type
        # Implicitly created VarPorts
        elif isinstance(port, ImplicitVarPort):
            return getattr(proc_model, port.var.name).d_type
        # Port has different name in Process and ProcessModel
        else:
            raise AssertionError("Port {!r} not found in "
                                 "ProcessModel {!r}".format(port, proc_model))

    # ToDo: (AW) Fix hard-coded hacks in this method and extend to other
    #  channel types
    def _create_channel_builders(self, proc_map: PROC_MAP) \
            -> ty.List[ChannelBuilderMp]:
        """Creates and returns ChannelBuilders which allows Runtime to build
        channels between Process ports.

        This method creates a ChannelBuilder for every connection from a
        source to a destination port in the graph of processes. OutPorts and
        RefPorts are considered source ports while InPorts and VarPorts are
        considered destination ports. A ChannelBuilder is only created for
        terminal connections from one leaf process to another. Intermediate
        ports of a hierarchical process are ignored.

        Note: Currently, this method does only support PyPyChannels.

        Once the Runtime has build the channel it can assign the
        corresponding CSP ports to the ProcessBuilder
        (i.e. PyProcBuilder.set_csp_ports(..)) and deploy the Process to the
        appropriate compute node.
        """
        ch_size = self._compile_config["pypy_channel_size"]
        channel_builders = []
        for src_p, src_pm in proc_map.items():
            # Get all source-like ports of source process
            src_ports = src_p.out_ports.members + src_p.ref_ports.members
            # Find destination ports for each source port
            for src_pt in src_ports:
                # Create PortInitializer for source port
                src_pt_dtype = self._get_port_dtype(src_pt, src_pm)
                src_pt_init = PortInitializer(
                    src_pt.name, src_pt.shape, src_pt_dtype,
                    src_pt.__class__.__name__, ch_size)
                # Create ChannelBuilder for all (src, dst) port pairs
                for dst_pt in src_pt.get_dst_ports():
                    # Get Process and ProcessModel of destination port
                    dst_p = dst_pt.process
                    if dst_p in proc_map:
                        # Only proceed if dst_pt is not just a dangling port
                        # on a hierarchical process
                        dst_pm = proc_map[dst_p]
                        # Find appropriate channel type
                        ch_type = self._get_channel_type(src_pm, dst_pm)
                        # Create PortInitializer for destination port
                        dst_pt_d_type = self._get_port_dtype(dst_pt, dst_pm)
                        dst_pt_init = PortInitializer(
                            dst_pt.name, dst_pt.shape, dst_pt_d_type,
                            dst_pt.__class__.__name__, ch_size)
                        # Create new channel builder
                        chb = ChannelBuilderMp(
                            ch_type, src_p, dst_p, src_pt_init, dst_pt_init)
                        channel_builders.append(chb)
                        # Create additional channel builder for every VarPort
                        if isinstance(dst_pt, VarPort):
                            # RefPort to VarPort connections need channels for
                            # read and write
                            rv_chb = ChannelBuilderMp(
                                ch_type, dst_p, src_p, dst_pt_init, src_pt_init)
                            channel_builders.append(rv_chb)

        return channel_builders

    # ToDo: (AW) Fix type resolution issues
    @staticmethod
    def _create_runtime_service_as_py_process_model(
            node_to_sync_domain_dict: ty.Dict[Node, ty.List[SyncDomain]],
            log: logging.getLoggerClass() = logging.getLogger()) \
            -> ty.Tuple[
                ty.Dict[SyncDomain, AbstractRuntimeServiceBuilder],
                ty.Dict[int, int]]:
        rs_builders: ty.Dict[SyncDomain, AbstractRuntimeServiceBuilder] = {}
        proc_id_to_runtime_service_id_map: ty.Dict[int, int] = {}
        rs_id: int = 0
        loihi_version: LoihiVersion = LoihiVersion.N3
        for node, sync_domains in node_to_sync_domain_dict.items():
            sync_domain_set = set(sync_domains)
            for sync_domain in sync_domain_set:
                if log.level == logging.DEBUG:
                    for resource in node.node_type.resources:
                        log.debug("node.node_type.resources: "
                                  + resource.__name__)
                if NeuroCore in node.node_type.resources:
                    rs_class = sync_domain.protocol.runtime_service[NeuroCore]
                elif Loihi1NeuroCore in node.node_type.resources:
                    log.debug("sync_domain.protocol. "
                              + "runtime_service[Loihi1NeuroCore]")
                    rs_class = sync_domain.protocol. \
                        runtime_service[Loihi1NeuroCore]
                    loihi_version: LoihiVersion = LoihiVersion.N2
                elif Loihi2NeuroCore in node.node_type.resources:
                    log.debug("sync_domain.protocol. "
                              + "runtime_service[Loihi2NeuroCore]")
                    rs_class = sync_domain.protocol. \
                        runtime_service[Loihi2NeuroCore]
                else:
                    rs_class = sync_domain.protocol.runtime_service[CPU]
                log.debug("RuntimeService Class: " + str(rs_class.__name__))
                model_ids: ty.List[int] = [p.id for p in sync_domain.processes]
                rs_builder = \
                    RuntimeServiceBuilder(rs_class=rs_class,
                                          protocol=sync_domain.protocol,
                                          runtime_service_id=rs_id,
                                          model_ids=model_ids,
                                          loihi_version=loihi_version)
                rs_builders[sync_domain] = rs_builder
                for p in sync_domain.processes:
                    proc_id_to_runtime_service_id_map[p.id] = rs_id
                rs_id += 1
        return rs_builders, proc_id_to_runtime_service_id_map

    def _create_mgmt_port_initializer(self, name: str) -> PortInitializer:
        return PortInitializer(
            name,
            (1,),
            np.float64,
            'MgmtPort',
            self._compile_config["pypy_channel_size"],
        )

    def _create_sync_channel_builders(
            self, rsb: ty.Dict[SyncDomain, AbstractRuntimeServiceBuilder]) \
            -> ty.Iterable[AbstractChannelBuilder]:
        sync_channel_builders: ty.List[AbstractChannelBuilder] = []
        for sync_domain in rsb:
            runtime_to_service = \
                RuntimeChannelBuilderMp(ChannelType.PyPy,
                                        Runtime,
                                        rsb[sync_domain],
                                        self._create_mgmt_port_initializer(
                                            f"runtime_to_service_"
                                            f"{sync_domain.name}"))
            sync_channel_builders.append(runtime_to_service)

            service_to_runtime = \
                RuntimeChannelBuilderMp(ChannelType.PyPy,
                                        rsb[sync_domain],
                                        Runtime,
                                        self._create_mgmt_port_initializer(
                                            f"service_to_runtime_"
                                            f"{sync_domain.name}"))
            sync_channel_builders.append(service_to_runtime)

            for process in sync_domain.processes:
                service_to_process = \
                    ServiceChannelBuilderMp(ChannelType.PyPy,
                                            rsb[sync_domain],
                                            process,
                                            self._create_mgmt_port_initializer(
                                                f"service_to_process_"
                                                f"{process.id}"))
                sync_channel_builders.append(service_to_process)

                process_to_service = \
                    ServiceChannelBuilderMp(ChannelType.PyPy,
                                            process,
                                            rsb[sync_domain],
                                            self._create_mgmt_port_initializer(
                                                f"process_to_service_"
                                                f"{process.id}"))
                sync_channel_builders.append(process_to_service)
        return sync_channel_builders

    def _create_exec_vars(self,
                          node_cfgs: ty.List[NodeConfig],
                          proc_map: PROC_MAP,
                          proc_id_to_runtime_service_id_map: ty.Dict[int, int]):

        for ncfg in node_cfgs:
            exec_vars = OrderedDict()
            for p, pm in proc_map.items():
                # Get unique Node and RuntimeService id
                node_id = ncfg.node_map[p].id
                run_srv_id: int = proc_id_to_runtime_service_id_map[p.id]
                # Create ExecVars for all Process Vars
                for v in p.vars:
                    if issubclass(pm, AbstractPyProcessModel):
                        ev = exec_var.PyExecVar(v, node_id, run_srv_id)
                    elif issubclass(pm, AbstractCProcessModel):
                        ev = exec_var.CExecVar(v, node_id, run_srv_id)
                    elif issubclass(pm, AbstractNcProcessModel):
                        ev = exec_var.PyExecVar(v, node_id, run_srv_id)
                    else:
                        raise NotImplementedError("Illegal ProcessModel type.")
                    exec_vars[v.id] = ev
            ncfg.set_exec_vars(exec_vars)

    def compile(self, proc: AbstractProcess, run_cfg: RunConfig) -> Executable:
        """Compiles a process hierarchy according to RunConfig and returns
        Executable which can either be serialized or executed by Runtime.

        The compiler proceeds through multiple stages to create an Executable
        for distributed heterogeneous HW platform:

        1. Extract specific ProcessModel classes for each Process in Process
        hierarchy according to RunCfg:
            - Parse hierarchical Process and either get sub process
            ProcessModel or leaf ProcessModel according to RunConfig
            - RunConfig may contain selection rules based on classes or
            specific class instances.
            - At this stage, also check if RunConfig selects any NcProcModels
            which would require access to NcProcCompiler and Loihi itself.
            Throw error if these modules are not found/installed -> raise
            NoLoihiProcessCompiler exception.
            - Returns a map of Process -> ProcessModel

        2. Group ProcessModel classes by ProcessModelType:
            - Returns a map of ProcessModelType -> {(Process, ProcessModel)}

        3. Compile ProcessModels by type:
            a) If any, compile NcProcessModels:
                - This is essentially the current NeuroProcCompiler except that
                we don't generate OutAx yet. That will be part of Channel setup
                - We can already allocate and account for OutAx resources but
                we cannot decide on specific addresses because those have not
                been determined yet.
                - Theoretically there could be mixtures of Loihi1 and Loihi2
                NeuroCores...
                - There is no reason to not also  include resource
                requirements to non-NcProcessModels (SeqProcs), i.e. not
                dealing with SpikeIO. We can already account for resource
                requirements but we fill in addresses later.
                - Currently, this would essentially build and return a board
                object but this could be wrapped into a uniform
                NcProcInitializer object for Runtime.

            b) If any, compile CProcessModels:
                - Check that uer provided behavior code exists (can't do much
                more than that at this point)
                - Generate header file code for Process-Vars given
                CProcessModel (but don't dump code to file yet)
                - Would there be any difference at this stage depending on
                which NodeType the process will run on?
                - For consistency, we could also return a CProcInitializer
                object for Runtime at this point that holds this loaded and
                generated code collateral

            c) If any, compile PyProcessModels: [Only part needed initially]
                - Gather shape information from lava.process ports and combine
                with LavaPyType (again, not much more we need to do here)
                - If we had fully working py2c tool, we could generate C code
                from Python code at this point
                - Return PyProcInitializer

        4. Generate NodeConfigurations:
            - The objective of this stage is to generate a list of
            different/alternate NodeConfigurations given resource
            requirements of all the processes. A NodeConfiguration is a set
            of Nodes, each supporting different ComputeResources and
            PeripheralResources of a certain quantity and each with a list of
            processes assigned to it.
            - Example: A GenericNode with a set of SeqProcs and a NahukuNode
            with a set of SeqProcs and NeuralProcs
            - Algorithm idea:
                - Initialize a binary matrix with rows corresponding to all
                processes and columns corresponding to different NodeTypes.
                - Set each element True, if the respective NodeType satisfies
                the Resource- and PeripheralType requirements of the Process
                - For M different NodeTypes, there are 2**M different node
                combinations. Not expected to be very large.
                - Search over all NodeType combinations:
                    - Convert index into binary vector
                    - Index into matrix columns
                    - Take OR over columns and sum the result
                    - Take those combinations that sum to the number of
                    processes
                - For all legal NodeType combinations, compute actual
                resource requirements given quantity of supported resources
                per node and exclude those that don't satisfy requirements of
                processes.
                - Feel free to compute some kind of score over remaining
                legal ones to thin out potential NodeCfgs even further
            - In order to potentially further prune the search space, consult
            RunConfiguration for excluded or required NodeTypes. Especially
            in the beginning, this will allows to bypass most of this search
            and just use the user-selected set of nodes.

        [From hereon, repeat everything for every remaining NodeCfg]

        5. Map processes:
            While we are still using NcCore, it will probably still take care
            of all of this.
            a) NcProcessModels: Map logical to physical core ids on-chip and
            map logical to physical chip ids on board given NodeCfg. To begin
            with we can continue to use our naive linear mapping strategy.
            b) CProcessModels: Map CProcessModels to one or multiple ECPU
            cores or CPU cores. We can have ParallelCProcess models to
            distribute a computation (if compatible) over multiple embedded
            cores.
            c) PyProcessModels: Not sure any mapping is needed.

        6. Setup RuntimeService processes:
            Given resource requirements from RunConfig (cluster size) and
            from lava.processModel compilation, the next step is to figure out
            where to instantiate auxiliary RuntimeService processes that
            handle synchronization, management access and probing.

            a) Instantiate RuntimeServiceInitializers per SyncDomain:
                - Iterate all SyncDomains
                    - Find Processes in SyncDomain
                    - Get all Nodes in current NodeCfg associated with those
                    Processes
                    - Create one or more RuntimeServiceInitializer (RSI) per
                    Node:
                        - Use some kind of factory pattern to create the
                        appropriate type of RSI give NodeType
                        - Add Processes on that Node to RSI
                        - On generic (Superhost-like) Node, one RSI is in
                        principle enough by we could create more for load
                        balancing as an optimization. Here RSI handles
                        management, synchronization and probing.
                        - On a LoihiNode, we need at least one RSI per host
                        CPU because it implies a shared memory domain in
                        which at least one RSI must handle management access.
                        But can also add more for load balancing. Here RSI
                        will have sub structure because synchronization happens
                        on ECPUs, management happens peer-to-peer and probing
                        happens from host.

            b) Compile probes:
                - For each RSI of a Node in a SyncDomain in a NodeCfg,
                get all Processes
                - Get probes of each process in RSI
                - Initialize those probes with address information produced
                during mapping stage
                - Register those probes with RSI directly.

        7. Compile channels:
            - At this point, all processes have been assigned to Nodes,
            ComputeResources and physical to logical mapping has been
            performed. Also RuntimeService processes have been created (or
            prepared for via RSIs).
            - This means we are ready to set up channels between all kinds of
            ports on different resources.
            - Channel implementations come in different forms:
                - Explicit channels under SW control, i.e. between two ports
                on a GenericNode CPU.
                - Implicit channels not under SW control, i.e. between two
                ports on LoihiCores or ECPUs. In this case, channel setup
                means updating OutAx to point to appropriate InAx and
                generating C header files with address information.
                - Mixtures of both across those two domains.

            a) Compile InPort/OutPort/RefPort channels (created by user):
                - Iterate all processes in current NodeCfg:
                    - Iterate all OutPorts and RefPorts:
                        - Get the other InPort or Var
                        - Create appropriate ChannelInitializer given type of
                        source and destination ProcessModel
                        - Assign channel to ports and collect all of them in
                        a global list

            b) Compile implied management channels (created by compiler):
                - Iterate all RSIs of a SyncDomain in a NodeCfg:
                    - Iterate all its assigned processes:
                        - Create management ports and channels on
                        RuntimeServiceProcess and process.
                            - The implementation of these ports and channels
                            again depends on type of source and destination
                            channels
                            - For LoihiNode we need to inform RSI about BS
                            domain size (currently inferred by NxCore)
                            - For non-Loihi nodes:
                                -  Create a MgmtRequest channel to process on
                                on which process can listen to sync signals
                                and management requests
                                - Create a MgmtResponse channel from
                                lava.process to any process that might send
                                management requests to it (i.e. via RefPorts,
                                handled in (a).

        Notes on Runtime (should probably go somewhere else):
            - Runtime can figure out what NodeTypes are available and pick a
              desired NodeConfiguration.
            - Given NodeConfiguration, Runtime will acquire required nodes
              (initializing nodes in compute cluster or acquiring Loihi board).
            - Next Runtime, will simply/naively execute every initializer
              provided by Executable. In many cases, this means sending the
              initializer to the remote node which instantiates the appropriate
              ProcessModel or RuntimeService and initializes internal state on
              the remote node.
              - For SW ProcessModels this means allocating e.g. numpy arrays
                and initializing their values
              - For LoihiNodes this basically means pushing the NxCore
                configuration until that gets eventually refactored.
        """

        # 1. Find all Processes hierarchically and select ProcessModel
        procs = self._find_processes(proc)
        for p in procs:
            if p._is_compiled:
                raise ex.ProcessAlreadyCompiled(p)
            p._is_compiled = True

        proc_map = self._map_proc_to_model(procs, run_cfg)
        # ToDo: Still need to register ProcModel somehow with Proc so
        #  interactive access is possible later

        # 2. Group Processes by ProcessModel type
        proc_groups = self._group_proc_by_model(proc_map)

        # 3. Compile ProcessModels by type
        exe = self._compile_proc_models(proc_groups)

        # 4. Create NodeConfigs (just pick one manually for now):
        node_cfgs = self._create_node_cfgs(proc_map, self.log)

        # 5. Create SyncDomains
        sync_domains, node_to_sync_domain_dict = self._create_sync_domains(
            proc_map, run_cfg, node_cfgs, self.log)

        # 6. Create Channel builders
        channel_builders = self._create_channel_builders(proc_map)

        # 7. Create Runtime Service builders
        runtime_service_builders, proc_id_to_runtime_service_id_map = \
            self._create_runtime_service_as_py_process_model(
                node_to_sync_domain_dict, self.log)

        # 8. Create ExecVars
        self._create_exec_vars(node_cfgs,
                               proc_map,
                               proc_id_to_runtime_service_id_map)

        # 9. Create Sync Channel Builders
        sync_channel_builders = self._create_sync_channel_builders(
            runtime_service_builders)

        # In the end, sd, nc and chb should be assigned to Executable so the
        # Runtime can access it.
        exe.set_sync_domains(sync_domains)
        exe.set_node_cfgs(node_cfgs)
        exe.set_rs_builders(runtime_service_builders)
        exe.set_channel_builders(channel_builders)
        exe.set_sync_channel_builders(sync_channel_builders)
        return exe
