# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import itertools
import logging
import os
import pickle  # noqa: S403 # nosec
import typing as ty
from collections import OrderedDict, defaultdict

import lava.magma.compiler.var_model as var_model
import numpy as np
from lava.magma.compiler.builders.interfaces import (AbstractChannelBuilder,
                                                     AbstractProcessBuilder)

try:
    from lava.magma.compiler.builders.c_builder import CProcessBuilder
    from lava.magma.compiler.builders.nc_builder import NcProcessBuilder
    from lava.magma.compiler.subcompilers.c.cproc_compiler import CProcCompiler
    from lava.magma.compiler.subcompilers.nc.ncproc_compiler import \
        NcProcCompiler
    from lava.magma.core.model.c.model import AbstractCProcessModel
    from lava.magma.core.model.nc.model import AbstractNcProcessModel
except ImportError:
    class CProcessBuilder(AbstractProcessBuilder):
        pass

    class NcProcessBuilder(AbstractProcessBuilder):
        pass

    class CProcCompiler:
        def __init__(self, *args, **kwargs):
            pass

    class NcProcCompiler:
        def __init__(self, *args, **kwargs):
            pass

    class AbstractCProcessModel:
        pass

    class AbstractNcProcessModel:
        pass

from lava.magma.compiler.builders.channel_builder import (
    RuntimeChannelBuilderMp, ServiceChannelBuilderMp)
from lava.magma.compiler.builders.runtimeservice_builder import \
    RuntimeServiceBuilder
from lava.magma.compiler.channel_map import ChannelMap, Payload, PortPair
from lava.magma.compiler.channels.interfaces import ChannelType
from lava.magma.compiler.compiler_graphs import ProcGroup, ProcGroupDiGraphs
from lava.magma.compiler.compiler_utils import split_proc_builders_by_type
from lava.magma.compiler.executable import Executable
from lava.magma.compiler.mapper import Mapper
from lava.magma.compiler.node import Node, NodeConfig
from lava.magma.compiler.subcompilers.channel_builders_factory import \
    ChannelBuildersFactory
from lava.magma.compiler.subcompilers.interfaces import AbstractSubCompiler
from lava.magma.compiler.subcompilers.py.pyproc_compiler import PyProcCompiler
from lava.magma.compiler.utils import PortInitializer
from lava.magma.core import resources
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import (CPU, LMT, Loihi1NeuroCore,
                                       Loihi2NeuroCore, NeuroCore)
from lava.magma.core.run_configs import RunConfig, AbstractLoihiHWRunCfg
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.runtime.runtime import Runtime
from lava.magma.runtime.runtime_services.enums import LoihiVersion
from lava.magma.compiler.channels.watchdog import WatchdogManagerBuilder


class Compiler:
    """Lava processes Compiler, called from any process in a process network.

    Creates an Executable for the network of processes connected to the
    process passed to the compile method.
    """

    def __init__(
        self,
        compile_config: ty.Optional[ty.Dict[str, ty.Any]] = None,
        loglevel: ty.Optional[int] = logging.WARNING,
    ):
        """Compiler that takes a network of Lava Processes and creates an
        Executable from it by creating and calling backend-specific
        SubCompilers.

        Parameters
        ----------
        compile_config : ty.Optional[ty.Dict[str, ty.Any]]
            Dictionary that may contain configuration options for the overall
            Compiler as well as all SubCompilers.
        loglevel : ty.Optional[int]
            Level of output to the log; default: 'logging.WARNING'.

        """
        self._compile_config = compile_config or {}
        self._compile_config.setdefault("loihi_gen", "oheogulch")
        self._compile_config.setdefault("pypy_channel_size", 64)
        self._compile_config.setdefault("long_event_timeout", 600.0)
        self._compile_config.setdefault("short_event_timeout", 60.0)
        self._compile_config.setdefault("use_watchdog", False)
        self.log = logging.getLogger(__name__)
        self.log.addHandler(logging.StreamHandler())
        self.log.setLevel(loglevel)

    def compile(
        self, process: AbstractProcess, run_cfg: RunConfig
    ) -> Executable:
        """Compiles all Processes connected to the given Process and the
        channels defined by their connectivity.

        Returns an Executable that contains all Builder instances required to
        execute the Lava Process network on heterogeneous hardware.

        Parameters
        ----------
        process : AbstractProcess
            Process from which all connected Processes in the Lava
            Process network are searched.
        run_cfg : RunConfig
            RunConfig that determines which ProcessModels will be selected
            for Processes.

        Returns
        -------
        executable : Executable
            An instance of an Executable that contains all required Builders.
        """
        # Group and sort all Processes connected to 'process' into a list of
        # ProcGroups.
        proc_group_digraph = ProcGroupDiGraphs(process,
                                               run_cfg,
                                               self._compile_config)
        proc_groups: ty.List[ProcGroup] = proc_group_digraph.get_proc_groups()
        # Get a flattened list of all AbstractProcesses
        process_list = list(itertools.chain.from_iterable(proc_groups))
        channel_map = ChannelMap.from_proc_groups(proc_groups)
        proc_builders, channel_map = self._compile_proc_groups(
            proc_groups, channel_map
        )
        _, c_builders, nc_builders = split_proc_builders_by_type(
            proc_builders
        )

        node_configs = self._create_node_cfgs(proc_groups)
        sync_domains, node_to_sync_domain_dict = self._create_sync_domains(
            proc_groups, run_cfg, node_configs, self.log
        )

        (
            runtime_service_builders,
            proc_id_to_runtime_service_id_map,
        ) = self._create_runtime_service_builder(
            node_to_sync_domain_dict,
            self.log,
            nc_builders,
            c_builders,
            run_cfg,
            self._compile_config,
        )
        channel_builders = ChannelBuildersFactory().from_channel_map(
            channel_map, compile_config=self._compile_config
        )
        sync_channel_builders = self._create_sync_channel_builders(
            runtime_service_builders
        )
        watchdog_manager_builder = \
            WatchdogManagerBuilder(self._compile_config,
                                   self.log.getEffectiveLevel())

        # Package all Builders and NodeConfigs into an Executable.
        executable = Executable(
            process_list,
            proc_builders,
            channel_builders,
            node_configs,
            sync_domains,
            runtime_service_builders,
            sync_channel_builders,
            watchdog_manager_builder
        )

        # Create VarModels.
        self._assign_nodecfg_rtservice_to_var_models(
            node_configs, proc_builders, proc_id_to_runtime_service_id_map
        )
        # Replace logical addresses with physical addresses.
        mapper = Mapper()
        mapper.map_cores(executable, channel_map)

        return executable

    def _compile_proc_groups(
        self, proc_groups: ty.List[ProcGroup], channel_map: ChannelMap
    ) -> ty.Tuple[
        ty.Dict[AbstractProcess, AbstractProcessBuilder],
        ty.Dict[PortPair, Payload],
    ]:
        """Compiles all Processes within all given ProcGroups with the
        respective SubCompiler they require.

        Parameters
        ----------
        proc_groups : ty.List[ProcGroup]
            A List of all ProcGroups to be compiled.
        channel_map : ChannelMap
            The global ChannelMap that contains information about Channels
            between Processes. This is used by the ChannelCompiler to
            generate ChannelBuilders, which are stored in the Executable.

        Returns
        -------
        proc_builders : ty.Dict[AbstractProcess, AbstractProcessBuilder]]
            A dictionary of builders for all Processes in all ProcGroups;
            this will be stored in the Executable.
        channel_map : ChannelMap
            The global dict-like ChannelMap given as input but with values
            updated according to partitioning done by subcompilers.
        """
        procname_to_proc_map: ty.Dict[str, AbstractProcess] = {}
        proc_to_procname_map: ty.Dict[AbstractProcess, str] = {}
        for proc_group in proc_groups:
            for p in proc_group:
                procname_to_proc_map[p.name] = p
                proc_to_procname_map[p] = p.name

        if self._compile_config.get("cache", False):
            cache_dir = self._compile_config["cache_dir"]
            if os.path.exists(os.path.join(cache_dir, "cache")):
                with open(os.path.join(cache_dir, "cache"), "rb") \
                        as cache_file:
                    cache_object = pickle.load(cache_file)  # noqa: S301 # nosec

                proc_builders_values = cache_object["procname_to_proc_builder"]
                proc_builders = {}
                for proc_name, pb in proc_builders_values.items():
                    proc = procname_to_proc_map[proc_name]
                    proc_builders[proc] = pb
                    pb.proc_params = proc.proc_params

                channel_map.read_from_cache(cache_object, procname_to_proc_map)
                print(f"\nBuilders and Channel Map loaded from "
                      f"Cache {cache_dir}\n")
                return proc_builders, channel_map

        # Get manual partitioning, if available
        partitioning = self._compile_config.get("partitioning", None)

        # Create the global ChannelMap that is passed between
        # SubCompilers to communicate about Channels between Processes.

        # List of SubCompilers across all ProcGroups.
        subcompilers = []

        for proc_group in proc_groups:
            # Create a mapping from SubCompiler class to a list of Processes
            # that must be compiled with that SubCompiler.
            subcompiler_to_procs = self._map_subcompiler_type_to_procs(
                proc_group
            )

            # Create all the SubCompiler instances required for this
            # ProcGroup and append them to the list of all SubCompilers.
            pg_subcompilers = self._create_subcompilers(subcompiler_to_procs)
            subcompilers.append(pg_subcompilers)

            # Compile this ProcGroup.
            self._compile_proc_group(pg_subcompilers, channel_map,
                                     partitioning)

        # Flatten the list of all SubCompilers.
        subcompilers = list(itertools.chain.from_iterable(subcompilers))

        # Extract Builders from SubCompilers.
        proc_builders, channel_map = self._extract_proc_builders(
            subcompilers, channel_map
        )

        if self._compile_config.get("cache", False):
            cache_dir = self._compile_config["cache_dir"]
            os.makedirs(cache_dir, exist_ok=True)
            cache_object = {}
            # Validate All Processes are Named
            procname_to_proc_builder = {}
            for p, pb in proc_builders.items():
                if p.name in procname_to_proc_builder or \
                        "Process_" in p.name:
                    msg = f"Unable to Cache. " \
                          f"Please give unique names to every process. " \
                          f"Violation Name: {p.name=}"
                    raise Exception(msg)
                procname_to_proc_builder[p.name] = pb
                pb.proc_params = None
            cache_object["procname_to_proc_builder"] = procname_to_proc_builder
            channel_map.write_to_cache(cache_object, proc_to_procname_map)
            with open(os.path.join(cache_dir, "cache"), "wb") as cache_file:
                pickle.dump(cache_object, cache_file)
            for p, pb in proc_builders.items():
                pb.proc_params = p.proc_params
            print(f"\nBuilders and Channel Map stored to Cache {cache_dir}\n")
        return proc_builders, channel_map

    @staticmethod
    def _map_subcompiler_type_to_procs(
        proc_group: ProcGroup,
    ) -> ty.Dict[ty.Type[AbstractSubCompiler], ty.List[AbstractProcess]]:
        """Returns a dictionary that maps the class of a SubCompiler to the
        list of all Processes within the given ProcGroup that must be
        compiled with that SubCompiler.

        Parameters
        ----------
        proc_group : ProcGroup
            A list of (possibly heterogeneous) Processes that must be sorted
            by their required SubCompiler types.

        Returns
        -------
        mapping : ty.Dict[ty.Type[AbstractSubCompiler],
                          ty.List[AbstractProcess]]
            A mapping from SubCompiler classes to a list of Processes within
            the given ProcGroup that must be compiled with them.
        """
        comp_to_procs = {}
        for proc in proc_group:
            model_cls = proc.model_class

            if issubclass(model_cls, AbstractPyProcessModel):
                comp_to_procs.setdefault(PyProcCompiler, []).append(proc)
            elif issubclass(model_cls, AbstractCProcessModel):
                comp_to_procs.setdefault(CProcCompiler, []).append(proc)
            elif issubclass(model_cls, AbstractNcProcessModel):
                comp_to_procs.setdefault(NcProcCompiler, []).append(proc)
            else:
                raise NotImplementedError(
                    "No subcompiler exists for the given ProcessModel of "
                    f"type {model_cls}"
                )

        return comp_to_procs

    def _create_subcompilers(
        self,
        compiler_type_to_procs: ty.Dict[
            ty.Type[AbstractSubCompiler], ty.List[AbstractProcess]
        ],
    ) -> ty.List[AbstractSubCompiler]:
        """Creates all SubCompiler instances that are required for the given
        mapping from SubCompiler to its Processes.

        Parameters
        ----------
        compiler_type_to_procs : ty.Dict[ty.Type[AbstractSubCompiler],
                                         ty.List[AbstractProcess]]
            A mapping from SubCompiler classes to a list of Processes within
            the given ProcGroup that must be compiled with them. Created by
            the method _map_compiler_type_to_procs().

        Returns
        -------
        subcompilers : ty.List[AbstractSubCompiler]
            A list of SubCompiler instances that have already been
            initialized with the Processes they will compile.
        """
        subcompilers = []
        c_idx = []
        nc_idx = []
        # Go through all required subcompiler classes...
        for idx, (subcompiler_class, procs) in \
                enumerate(compiler_type_to_procs.items()):
            # ...create the subcompiler instance...
            compiler = subcompiler_class(procs, self._compile_config)
            # ...and add it to the list.
            subcompilers.append(compiler)
            # Remember the index for C and Nc subcompilers:
            if isinstance(compiler, CProcCompiler):
                c_idx.append(idx)
            if isinstance(compiler, NcProcCompiler):
                nc_idx.append(idx)

        # Implement the heuristic "C-first Nc-second" whenever C and Nc
        # compilers are in the list. The subcompiler.compile is called in
        # their order of appearance in this list.
        # In the implementation below, we assume that there is only one
        # instance of each subcompiler (Py/C/Nc) per ProcGroup. This should
        # be true as long as the `compiler_type_to_procs` mapping comes from
        # `self._map_subcompiler_type_to_procs`.
        # 1. Confirm that there is only one instance of C and Nc subcompilers
        if len(c_idx) > 1 or len(nc_idx) > 1:
            raise AssertionError("More than one instance of C or Nc "
                                 "subcompiler detected.")
        # 2. If the index of C subcompiler is larger (appears later in the
        # list), then swap it with Nc subcompiler.
        if len(c_idx) > 0 and len(nc_idx) > 0:
            if c_idx[0] > nc_idx[0]:
                subcompilers[c_idx[0]], subcompilers[nc_idx[0]] = subcompilers[
                    nc_idx[0]], subcompilers[c_idx[0]]
        # We have ensured that C subcompiler appears before Nc subcompiler in
        # the list we return. As compile() is called serially on each
        # subcompiler, C Processes will be compiled before Nc Processes
        # within a ProcGroup.
        return subcompilers

    @staticmethod
    def _compile_proc_group(
        subcompilers: ty.List[AbstractSubCompiler], channel_map: ChannelMap,
        partitioning: ty.Dict[str, ty.Dict]
    ) -> None:
        """For a given list of SubCompilers that have been initialized with
        the Processes of a single ProcGroup, iterate through the compilation
        of all SubCompilers until the ChannelMap is no longer changed. The
        ChannelMap holds information about all channels that may span
        Processes compiled by different types of SubCompilers and may be
        updated by a call to AbstractSubCompiler.compile().

        Parameters
        ----------
        subcompilers : ty.List[AbstractSubCompiler]
            A list of SubCompilers.
        channel_map : ChannelMap
            The global ChannelMap that contains information about Channels
            between Processes.
        partitioning: ty.Dict
            Optional manual mapping dictionary used by ncproc compiler.
        """
        channel_map_prev = None

        # Loop until the SubCompilers have converged on how to distribute the
        # Process network onto the available compute resources. Convergence is
        # reflected in the ChannelMap no longer being updated by any
        # SubCompiler.
        while channel_map != channel_map_prev:
            channel_map_prev = channel_map.copy()
            for subcompiler in subcompilers:
                # Compile the Processes registered with each SubCompiler and
                # update the ChannelMap.
                channel_map = subcompiler.compile(channel_map, partitioning)

    @staticmethod
    def _extract_proc_builders(
        subcompilers: ty.List[AbstractSubCompiler], channel_map: ChannelMap
    ) -> ty.Tuple[ty.Dict[AbstractProcess, AbstractProcessBuilder], ChannelMap]:
        """Extracts all Builders from all SubCompilers and returns them in a
        dictionary.

        Parameters
        ----------
        subcompilers : ty.List[AbstractSubCompiler]
            list of SubCompilers that carry Builders of compiled Processes
        channel_map: A dictionary-like datastructure to share data among
        subcompilers.

        Returns
        -------
        proc_builders : ty.Dict[AbstractProcess, AbstractProcessBuilder]
            a dictionary of all Builders (of Processes) that were contained
            in the SubCompilers; each Builder is accessible in the dictionary
            by the Process that the Builder represents.
        """
        proc_builders = {}

        for subcompiler in subcompilers:
            builders, channel_map = subcompiler.get_builders(channel_map)
            if builders:
                proc_builders.update(builders)
        return proc_builders, channel_map

    def _create_node_cfgs(
        self, proc_groups: ty.List[ProcGroup]
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
        procs = list(itertools.chain.from_iterable(proc_groups))
        loihi_gen = self._compile_config["loihi_gen"]

        if self.log.level == logging.DEBUG:
            for proc in procs:
                self.log.debug(f"Proc Name: {proc.name}, Proc: {str(proc)}")
                self.log.debug(f"ProcModel: {str(proc.model_class)}")

        head_node = Node(node_type=resources.HeadNode, processes=[])

        # Until NodeConfig generation algorithm is present
        # check if NcProcessModel is present in proc_map,
        # if so add hardcoded Node for OheoGulch
        node_tracker = {}
        for proc in procs:
            if issubclass(
                proc.model_class, AbstractNcProcessModel
            ) or issubclass(proc.model_class, AbstractCProcessModel):
                self.log.debug("LOIHI_GEN: " + str(loihi_gen.upper()))
                if loihi_gen.upper() == resources.OheoGulch.__name__.upper():
                    if resources.OheoGulch not in node_tracker:
                        node = Node(
                            node_type=resources.OheoGulch, processes=[])
                        self.log.debug(
                            "OheoGulch Node Added to NodeConfig: "
                            + str(node.node_type)
                        )
                        node_tracker[resources.OheoGulch] = node
                    else:
                        node = node_tracker[resources.OheoGulch]
                    node.add_process(proc)
                elif loihi_gen.upper() == resources.Nahuku.__name__.upper():
                    if resources.Nahuku not in node_tracker:
                        node = Node(node_type=resources.Nahuku, processes=[])
                        self.log.debug(
                            "Nahuku Node Added to NodeConfig: "
                            + str(node.node_type)
                        )
                        node_tracker[resources.Nahuku] = node
                    else:
                        node = node_tracker[resources.Nahuku]
                    node.add_process(proc)
                else:
                    raise NotImplementedError("Not Supported")
            else:
                if resources.CPU not in node_tracker:
                    node_tracker[resources.CPU] = head_node
                head_node.add_process(proc)

        ncfg = NodeConfig()
        unique_nodes = set(list(node_tracker.values()))
        for n in unique_nodes:
            ncfg.append(n)

        return [ncfg]

    @staticmethod
    def _create_sync_domains(
        proc_groups: ty.List[ProcGroup],
        run_cfg: RunConfig,
        node_cfgs: ty.List[NodeConfig],
        log: logging.getLoggerClass(),
    ) -> ty.Tuple[ty.List[SyncDomain], ty.Dict[Node, ty.Set[SyncDomain]]]:
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
                pm = p.model_class

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
        for p in list(itertools.chain.from_iterable(proc_groups)):
            pm = p.model_class
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
                if not issubclass(pm, AbstractPyProcessModel):
                    default_sd_name = proto.__name__ + "_Nc_C" + "_SyncDomain"
                if default_sd_name in sync_domains:
                    # Default sync domain for current protocol already exists
                    sd = sync_domains[default_sd_name]
                else:
                    # ... or does not exist yet
                    sd = SyncDomain(name=default_sd_name, protocol=proto())
                    sync_domains[default_sd_name] = sd
                sd.add_process(p)
                proc_to_domain_map[p] = sd

        node_to_sync_domain_dict: ty.Dict[
            Node, ty.Set[SyncDomain]
        ] = defaultdict(set)
        for node_cfg in node_cfgs:
            for node in node_cfg:
                log.debug("Node: " + str(node.node_type.__name__))
                node_to_sync_domain_dict[node].update(
                    [proc_to_domain_map[proc] for proc in node.processes]
                )

        return list(sync_domains.values()), node_to_sync_domain_dict

    @staticmethod
    def _create_runtime_service_builder(
        node_to_sync_domain_dict: ty.Dict[Node, ty.Set[SyncDomain]],
        log: logging.getLoggerClass(),
        nc_builders: ty.Dict[AbstractProcess, NcProcessBuilder],
        c_builders: ty.Dict[AbstractProcess, CProcessBuilder],
        run_cfg: RunConfig,
        compile_config: ty.Optional[ty.Dict[str, ty.Any]] = None
    ) -> ty.Tuple[
        ty.Dict[SyncDomain, RuntimeServiceBuilder], ty.Dict[int, int]
    ]:
        rs_builders: ty.Dict[SyncDomain, RuntimeServiceBuilder] = {}
        proc_id_to_runtime_service_id_map: ty.Dict[int, int] = {}
        rs_id: int = 0

        loihi_gen = compile_config["loihi_gen"].upper()
        loihi_version: LoihiVersion = LoihiVersion.N3
        if loihi_gen == resources.OheoGulch.__name__.upper():
            loihi_version = LoihiVersion.N3
        if loihi_gen == resources.Nahuku.__name__.upper():
            loihi_version = LoihiVersion.N2

        for node, sync_domains in node_to_sync_domain_dict.items():
            sync_domain_set = sync_domains
            for sync_domain in sync_domain_set:
                node_resource_types = node.node_type.resources
                if log.level == logging.DEBUG:
                    for resource in node_resource_types:
                        log.debug(
                            "node.node_type.resources: " + resource.__name__
                        )

                if NeuroCore in node_resource_types:
                    rs_class = sync_domain.protocol.runtime_service[NeuroCore]
                elif LMT in node_resource_types:
                    rs_class = sync_domain.protocol.runtime_service[LMT]
                elif Loihi1NeuroCore in node_resource_types:
                    log.debug(
                        "sync_domain.protocol. "
                        + "runtime_service[Loihi1NeuroCore]"
                    )
                    rs_class = sync_domain.protocol.runtime_service[
                        Loihi1NeuroCore
                    ]
                    loihi_version: LoihiVersion = LoihiVersion.N2
                elif Loihi2NeuroCore in node_resource_types:
                    log.debug(
                        "sync_domain.protocol. "
                        + "runtime_service[Loihi2NeuroCore]"
                    )
                    rs_class = sync_domain.protocol.runtime_service[
                        Loihi2NeuroCore
                    ]
                else:
                    rs_class = sync_domain.protocol.runtime_service[CPU]
                log.debug("RuntimeService Class: " + str(rs_class.__name__))
                model_ids: ty.List[int] = [p.id for p in sync_domain.processes]

                rs_kwargs = {
                    "c_builders": list(c_builders.values()),
                    "nc_builders": list(nc_builders.values())
                }
                if isinstance(run_cfg, AbstractLoihiHWRunCfg):
                    rs_kwargs["callback_fxs"] = run_cfg.callback_fxs
                    rs_kwargs["embedded_allocation_order"] = \
                        run_cfg.embedded_allocation_order

                rs_builder = RuntimeServiceBuilder(
                    rs_class,
                    sync_domain.protocol,
                    rs_id,
                    model_ids,
                    loihi_version,
                    log.level,
                    compile_config,
                    **rs_kwargs
                )
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
            "MgmtPort",
            self._compile_config["pypy_channel_size"],
        )

    def _create_sync_channel_builders(
        self, rsb: ty.Dict[SyncDomain, RuntimeServiceBuilder]
    ) -> ty.Iterable[AbstractChannelBuilder]:
        sync_channel_builders: ty.List[AbstractChannelBuilder] = []
        for sync_domain in rsb:
            runtime_to_service = RuntimeChannelBuilderMp(
                ChannelType.PyPy,
                Runtime,
                rsb[sync_domain],
                self._create_mgmt_port_initializer(
                    f"runtime_to_service_" f"{sync_domain.name}"
                ),
            )
            sync_channel_builders.append(runtime_to_service)

            service_to_runtime = RuntimeChannelBuilderMp(
                ChannelType.PyPy,
                rsb[sync_domain],
                Runtime,
                self._create_mgmt_port_initializer(
                    f"service_to_runtime_" f"{sync_domain.name}"
                ),
            )
            sync_channel_builders.append(service_to_runtime)

            for process in sync_domain.processes:
                if issubclass(process.model_class, AbstractPyProcessModel):
                    service_to_process = ServiceChannelBuilderMp(
                        ChannelType.PyPy,
                        rsb[sync_domain],
                        process,
                        self._create_mgmt_port_initializer(
                            f"service_to_process_" f"{process.id}"
                        ),
                    )
                    sync_channel_builders.append(service_to_process)

                    process_to_service = ServiceChannelBuilderMp(
                        ChannelType.PyPy,
                        process,
                        rsb[sync_domain],
                        self._create_mgmt_port_initializer(
                            f"process_to_service_" f"{process.id}"
                        ),
                    )
                    sync_channel_builders.append(process_to_service)
        return sync_channel_builders

    def _assign_nodecfg_rtservice_to_var_models(
        self,
        node_cfgs: ty.List[NodeConfig],
        proc_builders: ty.Dict[AbstractProcess, AbstractProcessBuilder],
        proc_id_to_runtime_service_id_map: ty.Dict[int, int],
    ):

        # Type def
        VvMap = ty.Dict[
            int,
            ty.Union[
                var_model.AbstractVarModel,
                var_model.PyVarModel,
                var_model.CVarModel,
                var_model.NcVarModel,
            ],
        ]
        for ncfg in node_cfgs:
            var_models: VvMap = {}
            for p, pb in proc_builders.items():
                # Note: For NcProcesses, not all Processes are recorded in
                #  proc_builders dictionary. Only the compartment-like
                #  Processes, which are central to forming NeuroProcGroups
                #  are recorded. But these should share the same NodeID and
                #  RuntimeServiceID with the rest of the NcProcesses within
                #  the same NeuroProcGroup.
                # Get unique Node and RuntimeService id
                node_id = ncfg.node_map[p].id
                run_srv_id: int = proc_id_to_runtime_service_id_map[p.id]
                # Assign node and runtime service ids to each varmodel
                # within this ProcessBuilder
                for vm in pb.var_id_to_var_model_map.values():
                    vm.node_id = node_id
                    vm.runtime_srv_id = run_srv_id
                var_models.update(pb.var_id_to_var_model_map)
            ncfg.set_var_models(var_models)
