# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import importlib
import importlib.util as import_utils
import inspect
import itertools
import os
import pkgutil
import sys
import types
import typing as ty
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum, auto
import warnings

import lava.magma.compiler.exceptions as ex
import networkx as ntx
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel

try:
    from lava.magma.core.model.c.model import AbstractCProcessModel
    from lava.magma.core.model.nc.model import AbstractNcProcessModel
except ImportError:
    class AbstractCProcessModel:
        pass

    class AbstractNcProcessModel:
        pass

from lava.magma.core.process.ports.ports import (AbstractPort,
                                                 AbstractVirtualPort, InPort,
                                                 OutPort, RefPort, VarPort)
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.run_configs import RunConfig

ProcMap = ty.Dict[AbstractProcess, ty.Type[AbstractProcessModel]]
ProcGroup = ty.List[AbstractProcess]


class ProcessModelTypes(Enum):
    """Enumeration of different types of ProcessModels: Py, C, Nc, etc.
    """
    PY = AbstractPyProcessModel
    C = AbstractCProcessModel
    NC = AbstractNcProcessModel


class NodeAnnotation(Enum):
    """Annotations for the nodes of directed graphs created for compilation.
    The node attribute called "degdef" (short for "degree deficit") of a
    node takes one of the following seven values:

    1. PUREIN, a pure input Process: in degree of the node = 0
    2. PUREOUT a pure output Process: out degree of the node = 0
    3. ISOLATED a Process node with in degree = out degree = 0
    4. INLIKE  an in-like Process: (out degree) - (in degree) < 0
    5. OUTLIKE an out-like Process: (out degree) - (in degree) > 0
    6. NEUTRAL a Process node with in degree = out degree
    7. INVALID a Process node that does not fit any of the above (unlikely)

    """
    PUREIN = auto()
    PUREOUT = auto()
    ISOLATED = auto()
    INLIKE = auto()
    OUTLIKE = auto()
    NEUTRAL = auto()
    INVALID = auto()


def flatten_list_recursive(ll: ty.List) -> ty.List:
    """Recursively flatten a list of lists.

    Parameters
    ----------
    ll : list
        Any list of lists (of any depth)

    Returns
    -------
    ll : list
        Flattened list

    Notes
    -----
    Taken from: https://stackabuse.com/python-how-to-flatten-list-of-lists/
    """
    if len(ll) == 0:
        return ll
    if isinstance(ll[0], list):
        return flatten_list_recursive(ll[0]) + flatten_list_recursive(ll[1:])
    return ll[:1] + flatten_list_recursive(ll[1:])


def flatten_list_itertools(ll: ty.List) -> ty.List:
    """Simpler way to flatten nested lists.
    """
    return list(itertools.chain.from_iterable(ll))


def find_processes(proc: AbstractProcess,
                   seen_procs: ty.List[AbstractProcess] = None) -> \
        ty.List[AbstractProcess]:
    """Find all processes that are connected to `proc`.

    Processes are connected via different kinds of Ports to other
    processes. This method starts at the given `proc` and traverses the
    graph along connections to find all connected processes.

    During compilation, this method is called **before** discovering
    ProcessModels implementing Processes.

    Parameters
    ----------
    proc : AbstractProcess
        Base process starting which the discovery of all connected Processes
        begins.
    seen_procs : List[AbstractProcess]
        A list of Processes visited during traversal of connected Processes.
        Used for making the method recursive. This parameter is set to
        `None` at the time of the first call.

    Returns
    -------
    seen_procs : List[AbstractProcess]
        A list of all discovered Processes.
    """

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
            new_list.extend(
                find_processes(proc, seen_procs))

    seen_procs.extend(new_list)
    seen_procs = list(set(seen_procs))

    return seen_procs


def annotate_folded_view(proc_list: ty.List[AbstractProcess],
                         folded_procs: ty.List[str] = None):
    """Annotate folded views and propagate them recursively
    """
    annotated : ty.Set(AbstractProcess) = set()
    fv_inst_id : int = 0
    for p in proc_list:
        if p.__class__.__name__ in folded_procs:
            p.folded_view = p.__class__
            p.folded_view_inst_id = fv_inst_id
            fv_inst_id += 1
            annotated.add(p)

    for p in annotated:
        p.propagate_folded_views()


class DiGraphBase(ntx.DiGraph):
    """Base class for directed graphs in the compiler.

    The nodes of instances of this class can be any hashable objects,
    they need not be of type AbstractProcess. Inherits from NetworkX.DiGraph
    class.
    """

    def __init__(self, *args, **kwargs):
        super(DiGraphBase, self).__init__(*args, **kwargs)
        self.annotate_digraph_by_degree()

    def annotate_digraph_by_degree(self):
        """Annotate the graph's nodes according to their degree.

        More specifically, the annotations are based on the degree deficit
        of a node, defined as (out degree - in degree).

        See Also
        --------
        NodeAnnotation enum
        """
        # Create a new empty node attribute called 'degdef' (degree deficit)
        ntx.set_node_attributes(self, "", name="degdef")

        for node, nodeattr in self.nodes.items():
            if self.out_degree[node] == 0:
                if self.out_degree[node] == self.in_degree[node]:
                    nodeattr['degdef'] = NodeAnnotation.ISOLATED
                else:
                    nodeattr['degdef'] = NodeAnnotation.PUREOUT
            elif self.in_degree[node] == 0:
                nodeattr['degdef'] = NodeAnnotation.PUREIN
            elif self.out_degree[node] - self.in_degree[node] > 0:
                nodeattr['degdef'] = NodeAnnotation.INLIKE
            elif self.out_degree[node] - self.in_degree[node] < 0:
                nodeattr['degdef'] = NodeAnnotation.OUTLIKE
            elif self.out_degree[node] == self.in_degree[node]:
                nodeattr['degdef'] = NodeAnnotation.NEUTRAL
            else:
                nodeattr['degdef'] = NodeAnnotation.INVALID

    def is_dag(self, graph: 'DiGraphBase' = None) -> ty.Tuple[bool,
                                                              'DiGraphBase']:
        """Check if the input DiGraphBase is a DAG by recursive leaf pruning.

        Parameters
        ----------
        graph : DiGraphBase
            This parameter is only needed for recursion. It holds the residual
            graph after pruning leaf nodes of the graph from previous
            recursive step. At the first iterations, `graph = None`.

        Returns
        -------
        DAGness of the graph: Bool
            True if the graph is a DAG, else False
        graph : DiGraphBase
            The residual subgraph that remains after the input graph runs
            out of leaf nodes after all recursions are over.
        """

        if graph is None:
            graph = self.copy()

        graph.annotate_digraph_by_degree()

        leaves = [node for node, nodeattr in graph.nodes.items() if
                  nodeattr['degdef'] == NodeAnnotation.PUREIN]
        num_leaves = len(leaves)

        if num_leaves > 0:
            # Prune leaves
            graph.remove_nodes_from(leaves)
            _, graph = self.is_dag(graph)

        if len(list(graph.nodes)) > 1:
            isolated_nodes = \
                [node for node, nodeattr in graph.nodes.items() if nodeattr[
                    'degdef'] == NodeAnnotation.ISOLATED]
            if len(list(graph.nodes)) == len(isolated_nodes):
                return True, graph
            return False, graph
        else:
            return True, graph

    def collapse_subgraph_to_node(self, subgraph: 'DiGraphBase') -> \
            'DiGraphBase':
        """Replace any connected subgraph of the DiGraphBase object with a
        single node, while preserving connectivity.

        The new node is a DiGraphBase object, which preserves the internal
        connectivity of the subgraph.

        Parameters
        ----------
        subgraph : DiGraphBase
            Subgraph which needs to be collapsed into a single node. It needs
            to be connected..

        Returns
        -------
        out_graph : DiGraphBase
            A copy of `self` with node-replacement surgery complete.
        """

        if not set(self.nodes).intersection(set(subgraph.nodes)) == set(
                subgraph.nodes):
            raise AssertionError("The set of nodes of input subgraph is not a "
                                 "proper subset of nodes of the graph.")

        out_graph = self.copy()
        # If the subgraph contains 0 nodes, there is nothing to
        # collapse/condense. Just return the original graph.
        if len(list(subgraph.nodes)) < 1:
            return out_graph
        # If the subgraph contains onlt 1 node AND it has degree 0, there is
        # nothing to collapse/condense. Just return the original graph.
        if len(list(subgraph.nodes)) == 1 and \
                list(subgraph.nodes.values())[0]['degdef'] == \
                NodeAnnotation.ISOLATED:
            return out_graph
        # If the subgraph is the same as the main graph, remove
        # everything from out_graph, add just 1 node corresponding to the
        # entire graph, annotate the new graph, and return.
        if list(subgraph.edges) == list(self.edges) and set(subgraph.nodes) \
                == set(self.nodes):
            out_graph.remove_nodes_from(list(out_graph.nodes))
            out_graph.add_node(self)
            out_graph.annotate_digraph_by_degree()
            return out_graph

        # The following algorithm is based on:
        #     https://stackoverflow.com/a/35109402
        # For each node in subgraph,
        #   1. all its neighbours are identified,
        #   2. those neighbours that are already in the subgraph are ignored,
        #   3. those neighbours not in the subgraph are directly connected to
        #      the subgraph. The entire subgraph is treated as a node in the
        #      main graph.
        #   4. nodes belonging to the subgraph are deleted from the main graph
        for node in subgraph:
            neighbours = set(out_graph.successors(node)).union(set(
                out_graph.predecessors(node)))
            for neighbour in neighbours - set(subgraph):
                if neighbour in set(out_graph.successors(node)):
                    out_graph.add_edge(subgraph, neighbour)
                if neighbour in set(out_graph.predecessors(node)):
                    out_graph.add_edge(neighbour, subgraph)
            out_graph.remove_node(node)

        out_graph.annotate_digraph_by_degree()
        return out_graph

    def collapse_cycles_to_nodes(self) -> 'DiGraphBase':
        """Find simple cycles and collapse them on a single node iteratively,
        until no simple cycles can be found or the entire graph reduces to a
        single simple cycle.

        Returns
        -------
        out_graph : DiGraphBase
            A copy of `self` with node-replacement surgery complete.
        """

        # Create a working copy of the graph
        out_graph = self.copy()
        while True:
            # Detect simple cycles in out_graph
            cycle_list = list(ntx.simple_cycles(out_graph))
            if len(cycle_list) < 1:
                # If there are no simple cycles in the end
                break
            # Find the longest simple cycle
            lc = cycle_list[0]
            for c in cycle_list:
                if len(c) > len(lc):
                    lc = c
            # Subgraph corresponding to the longest simple cycle
            sglc = out_graph.subgraph(lc).copy()
            # Collapse the longest cycle to a node
            out_graph = out_graph.collapse_subgraph_to_node(sglc)

        return out_graph


class ProcDiGraph(DiGraphBase):
    """Directed graph data structure for the compiler which has some nodes
    that are Lava Processes.

    If a list of Lava Processes is passed to the constructor,
    using `proc_graph` keyword argument, a DiGraph will be created with the
    Processes as nodes and their interconnectivity as edges.

    Inherits from DiGraphBase, which in turn inherits from NetworkX.DiGraph.
    If the constructor is passed `*args` and `**kwargs` that create parts of a
    NetworkX.DiGraph, then the graph with Process nodes is created in
    addition to (and disjoint from) the existing graph elements.
    """

    def __init__(self, *args, **kwargs):

        proc_list = kwargs.pop("proc_list", None)
        super(ProcDiGraph, self).__init__(*args, **kwargs)

        if proc_list:
            self.add_nodes_from(proc_list)
            node_name_dict = dict(
                zip(proc_list, [proc.name for proc in proc_list]))
            ntx.set_node_attributes(self, node_name_dict, name="name")

            for proc in proc_list:
                # Get all processes connected in to and connected out from proc
                in_proc_list, out_proc_list = \
                    ProcDiGraph._traverse_ports_of_proc(proc)

                in_edge_list = list(
                    zip(in_proc_list, [proc] * len(in_proc_list)))
                self.add_edges_from(in_edge_list)

                out_edge_list = list(
                    zip([proc] * len(out_proc_list), out_proc_list))
                self.add_edges_from(out_edge_list)

            self.annotate_digraph_by_degree()

    @staticmethod
    def _get_port_direction(port: AbstractPort) -> str:
        """Determines the connectivity direction of a Port.

        "inward" for InPort and VarPort
        "outward" for OutPort and RefPort

        Parameters
        ----------
        port : AbstractPort
            The port of which connectivity direction is determined

        Returns
        -------
        port_dir : str
            The connectivity direction of `port`
        """
        if isinstance(port, InPort) or isinstance(port, VarPort):
            return "inward"
        elif isinstance(port, OutPort) or isinstance(port, RefPort):
            return "outward"
        elif isinstance(port, AbstractVirtualPort):
            return ProcDiGraph._get_port_direction(port._parent_port)
        else:
            raise TypeError(f"Invalid port type: {port}.")

    @staticmethod
    def _find_terminal_procs_recursively(port_list: ty.List[AbstractPort],
                                         trace_dir: str) -> \
            ty.List[AbstractProcess]:
        """Iterate through a list of ports of a Process and for each port in
        the list, recursively follow ports in a specified tracing/traversal
        direction until a terminal process is reached.

        Parameters
        ----------
        port_list : List[AbstractPort]
            A list of ports to iterate through and recursively follow.
        trace_dir : str
            The specified trace/traversal direction. Allowed values are
            "inward" or "outward".

        Returns
        -------
        proc_list : List[AbstractProcess]
            A list of all terminal/leaf processes.
        """

        proc_list = []
        if trace_dir != "inward" and trace_dir != "outward":
            raise ValueError("Invalid traversal direction. Must be 'inward' "
                             "or 'outward'.")
        # in_connections or out_connections attributes of ports
        collection_attr = trace_dir[:-4] + '_connections'
        for port in port_list:
            port_dir = ProcDiGraph._get_port_direction(port)
            conn_collection = getattr(port, collection_attr)
            for conn in conn_collection:
                conn_dir = ProcDiGraph._get_port_direction(conn)
                # If there is a connection to follow in the same direction,
                # follow it.
                if len(getattr(conn, collection_attr)) > 0:
                    proc_list.extend(
                        ProcDiGraph._find_terminal_procs_recursively([conn],
                                                                     trace_dir))
                # If there is nothing to follow, we have reached a 'terminal'
                # Port/connection. However, if it is of same type as the
                # previous Port *and* same type as the traversal direction,
                # then we have hit the "wall" of a hierarchical process. For
                # example, traversing "outward", if port and conn are
                # both OutPorts, and len(conn.out_connections) == 0. We
                # ignore/skip such a terminal Port and continue the loop.
                elif port_dir == conn_dir and port_dir == trace_dir:
                    continue
                # Finally, we reach a legitimate terminal port, with no
                # further depth to follow. We add the parent process of conn
                # to proc list.
                else:
                    proc_list.append(conn.process)

        return proc_list

    @staticmethod
    def _traverse_ports_of_proc(proc: AbstractProcess) -> ty.Tuple[
            ty.List[AbstractProcess], ty.List[AbstractProcess]]:
        """Traverse along port connectivity of all ports of the input Process.

        The SubProcessModels of hierarchical Processes are expected to be
        resolved and built. The traversal treats hierarchical Processes as
        transparent/pass-through. Port-connectivity is followed recursively
        till it reaches a terminal leaf Processes.

        Parameters
        ----------
        proc : AbstractProcess
            The 'base' process, whose port-connectivity needs to be traversed.

        Returns
        -------
        in_proc_list : List[AbstractProcess]
            A list of Processes connected *to* the input Process
        out_proc_list : List[AbstractProcess]
            A list of Processes connected *from* the input Process.
        """

        # Ports with incoming connections
        port_list_in = proc.in_ports.members + proc.var_ports.members
        in_proc_list = \
            ProcDiGraph._find_terminal_procs_recursively(
                port_list_in, "inward")
        # Ports with outgoing connections
        port_list_out = proc.out_ports.members + proc.ref_ports.members
        out_proc_list = \
            ProcDiGraph._find_terminal_procs_recursively(port_list_out,
                                                         "outward")

        return in_proc_list, out_proc_list

    def convert_to_procid_graph(self) -> DiGraphBase:
        """Convert ProcDiGraph to a DiGraph whose nodes are just the process
        ids of Processes from the original ProcDiGraph.

        This utility method is useful to compare two ProcDiGraphs without
        actually using entire AbstractProcess objects as nodes.

        Returns
        -------
        procid_graph : DiGraphBase
            A graph with nodes = process ids of Processes forming nodes of
            ProcDiGraph. Each node retains the attributes of the original nodes.
        """
        nodelist = [proc.id for proc in self.nodes]
        nodeattrdict = {proc.id: data for proc, data in self.nodes.items()}
        procid_graph = DiGraphBase()
        procid_graph.add_nodes_from(nodelist)
        ntx.set_node_attributes(procid_graph, nodeattrdict)

        edges_orig = list(self.edges)
        edgelist = []
        for (src_proc, dst_proc) in edges_orig:
            new_edge = (src_proc.id, dst_proc.id)
            edgelist.append(new_edge)
        procid_graph.add_edges_from(edgelist)

        return procid_graph


class AbstractProcGroupDiGraphs(ABC):
    """Abstract class for generating and holding directed graphs needed to
    create ProcGroups for compilation.

    Concrete subclass must implement `get_proc_groups()` method that outputs a
    list of ProcGroups, where `type(ProcGroup) = List[AbstractProcess]`.

    Downstream compiler iterates over the list of ProcGroups and invokes
    appropriate sub-compilers depending on the ProcessModel type.
    """
    @abstractmethod
    def get_proc_groups(self) -> ty.List[ProcGroup]:
        pass


class ProcGroupDiGraphs(AbstractProcGroupDiGraphs):
    """Concrete subclass of `AbstractProcGroupDiGraphs` that generates
    and holds various directed graphs needed to generate ProcGroups needed
    for compilation.

    All sub-compilers need the following tasks to be complete before they are
    instantiated:

    1. Find Processes:
        Using the "base" Process, on which proc.run(...) was
        called, a list of all processes connected to the base process is
        generated by traversing all InPort and OutPort connectivity.
    2. Generate RawProcDiGraph:
        Using the list in step 1, a directed graph is generated with
        Processes as nodes. As the exact behaviour of Processes is not
        known (i.e., ProcessModels), the directed graph is 'raw',
        without resolving any hierarchical Processes.
    3. Find and select ProcessModels:
        Using a RunConfig along with the list of Processes in step 1,
        a dictionary mapping Process to ProcessModel is created. Then
        ProcessModels are assigned as attributes to the Process instances.
    4. Generate ResolvedProcDiGraph:
        Using the dict mapping Processes to ProcessModels, create a
        directed graph with Processes as nodes. By this point,
        hierarchical Processes are known and resolved using their
        SubProcessModels (therefore the name). As these Processes do
        not feature in the dict mapping, they do not constitute a node in
        the graph.
    5. Generate IsoModelCondensedDiGraph:
        Using the ResolvedProcDiGraph generated in step 4, find all
        Process nodes with the same ProcessModel type, which are
        neighbours of each other (in the sense of Graph Theory), and
        collapse/condense them into a single node, while preserving the
        connectivity.
    6. Generate ProcGroupDiGraph:
        Using the IsoModelCondensedDiGraph from step 5, find all node
        that are not connected via feed-forward topology (i.e., all nodes
        that are a part of some cyclic structure) and collapse/condense
        them into a single node, while preserving connectivity. Each node
        of ProcGroupDiGraph thus generated is a ProcGroup.

    The ProcGroupDiGraph, by construction, is a single simple cycle or a
    DAG. In the first case, the simple cycle is broken at an arbitrary
    edge and List[ProcGroup] is generated from the list of its nodes.
    In the case that ProcGroupDiGraph is a DAG, it is ordered using
    topological sorting and List[ProcGroup] is produced by reversing the
    sorted order of nodes.
    """

    def __init__(self, proc: AbstractProcess, run_cfg: RunConfig,
                 compile_config: ty.Optional[ty.Dict[str, ty.Any]] = None):

        self._base_proc = proc  # Process on which compile/run was called
        self._run_cfg = run_cfg
        self._compile_config = compile_config
        # 1. Find all Processes
        proc_list = find_processes(proc)

        # Check if any Process in proc_list is already compiled
        for p in proc_list:
            if p.is_compiled:
                raise ex.ProcessAlreadyCompiled(p)
            p._is_compiled = True

        # Number of Processes before resolving HierarchicalProcesses
        self._num_procs_pre_pm_discovery = len(proc_list)
        # 2. Generate a ProcessGraph: This does not resolve
        # HierarchicalProcesses yet, because we have not discovered
        # ProcessModels
        self._raw_proc_digraph = ProcDiGraph(proc_list=proc_list)
        # 3. Find and select ProcessModels based on RunConfig:
        elab_procs = []
        proc_procmodel_map = ProcGroupDiGraphs._map_proc_to_model(proc_list,
                                                                  self._run_cfg,
                                                                  elab_procs)
        if self._compile_config and 'folded_view' in self._compile_config:
            folded_views = self._compile_config["folded_view"]
            annotate_folded_view(elab_procs, folded_views)

        # Assign ProcessModels to Processes
        for p, pm in proc_procmodel_map.items():
            p._model_class = pm
        # Number of processes after resolving HierarchicalProcesses
        self._num_procs_post_sub_exp = len(list(proc_procmodel_map.keys()))
        # 4. Generate ResolvedProcDiGraph: In the dictionary {Process:
        # ProcessModel}, SubProcessModels are already resolved. Therefore,
        # ResolvedProcDiGraph will not contain HierarchicalProcesses.
        # First create a list of Processes from the dict/map above
        resolved_proc_list = \
            [proc for proc in list(proc_procmodel_map.keys())]
        # Generate a graph with Processes as nodes
        self._resolved_proc_digraph = ProcDiGraph(proc_list=resolved_proc_list)
        # 5. Generate IsoModelCondensedDiGraph: a graph after condensing
        # connected Processes with same type of ProcessModel into a single node
        self._isomodel_condensed_digraph = self._collapse_isomodel_procs()
        # 6. Generate ProcGroupDiGraph: a graph after condensing cycles
        # into a single node; always either a DAG or a simple cycle by
        # construction, compilation order is derived from this graph.
        self._proc_group_digraph = \
            self._isomodel_condensed_digraph.collapse_cycles_to_nodes()
        # Check if self._compile_group_graph is DAG or is itself a simple cycle.
        chk1, _ = self._proc_group_digraph.is_dag()
        chk2 = len(list(ntx.simple_cycles(self._proc_group_digraph))) == 1
        if not chk1 and not chk2:
            raise NotImplementedError("LavaProcGraph init failed. The graph "
                                      "generated as proc_group_digraph is "
                                      "neither a DAG nor a simple cycle. "
                                      "Does the original computational graph "
                                      "of Processes contain disconnected "
                                      "components?")

    @property
    def raw_proc_digraph(self):
        """Directed graph of all connected Processes _before_ ProcessModel
        discovery.

        This graph is generated using Process-level connectivity before any
        ProcessModels are discovered. Any sub-Processes inside a
        SubProcessModel of a hierarchical Process will not show up in this
        graph.
        """
        return self._raw_proc_digraph

    @property
    def resolved_proc_digraph(self):
        """Directed graph of all connected Processes _after_ ProcessModel
        discovery.

        This graph is generated after discovering all ProcessModels and
        resolving/building SubProcessModels. Therefore, no hierarchical
        Processes show up in this graph. They are substituted by the
        sub-Processes inside their SubProcessModels.
        """
        return self._resolved_proc_digraph

    @property
    def isomodel_condensed_digraph(self):
        """Directed graph derived from `ProcGroupDiGraphs.resolved_proc_graph`,
        such that all *connected* Processes with same type of ProcessModel
        are collapsed/condensed onto a single node in the graph.
        """
        return self._isomodel_condensed_digraph

    @property
    def proc_group_digraph(self):
        """Directed graph derived from
        `ProcGroupDiGraphs.isomodel_condensed_graph`, such that all nodes
        involved in any kind of non-linear/feedback/cyclical topology are
        collapsed/condensed onto a single node.

        A proc_group_digraph - by construction - is a single simple cycle or
        a DAG.
        """
        return self._proc_group_digraph

    @property
    def base_proc(self):
        """The Process on which Process.run(...) is called.
        """
        return self._base_proc

    @property
    def num_procs_pre_pm_discovery(self):
        """Number of Processes before ProcessModel discovery.

        This is the number of nodes in `raw_proc_digraph`.

        See also
        --------
        raw_proc_digraph
        """
        return self._num_procs_pre_pm_discovery

    @property
    def num_procs_post_subproc_expansion(self):
        """Number of leaf Processes, after ProcessModel discovery and
        SubProcessModel resolution/expansion.

        This is the number of nodes in `resolved_proc_digraph`

        See also
        --------
        resolved_proc_digraph
        """
        return self._num_procs_post_sub_exp

    @staticmethod
    def _find_proc_models_in_module(proc: AbstractProcess,
                                    module: types.ModuleType) \
            -> ty.List[ty.Type[AbstractProcessModel]]:
        """Find and return all ProcModels that implement given Process in
        given Python module.

        Parameters
        ----------
        proc : AbstractProcess
            The Process for which ProcessModels are to be found.
        module : types.ModuleType
            The Python module in which the ProcessModels are searched.

        Returns
        -------
        proc_models : List[AbstractProcessModel]
            A list of all ProcessModels discovered in `module` that
            implement the behaviour of `proc`.
        """

        proc_models = []
        classes = [cls for cls in module.__dict__.values()
                   if inspect.isclass(cls)
                   and cls.__module__ == module.__name__]
        for cls in classes:
            if (
                    hasattr(cls, "implements_process")
                    and issubclass(cls, AbstractProcessModel)
                    and cls.implements_process is not None
                    and type(proc) is cls.implements_process
            ):
                # Collect ProcessModel
                proc_models.append(cls)

        return proc_models

    @staticmethod
    def _find_proc_models(proc: AbstractProcess) \
            -> ty.List[ty.Type[AbstractProcessModel]]:
        """Find all ProcessModels that implement given Process.

        First, we search in the same Python module, in which 'proc' is defined,
        for ProcModels implementing 'proc'. Next, we search through modules in
        the same directory as well as analogous directories of other Lava
        repositories listed in the PYTHONPATH.

        Parameters
        ----------
        proc : AbstractProcess
            The Process for which ProcessModels need to be found.

        Returns
        -------
        proc_models : List[AbstractProcessModels]
            A list of all ProcessModels that implement the behaviour of `proc`.
        """

        # Find all ProcModel classes that implement 'proc' in same module
        proc_module = sys.modules[proc.__module__]
        proc_models = \
            ProcGroupDiGraphs._find_proc_models_in_module(proc, proc_module)

        # Search for the file of the module.
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

        # Find all ProcModel classes that implement 'proc' in the same
        # directory and namespace module.

        dir_names = [os.path.dirname(file)]

        if not proc_module.__name__ == "__main__":
            # Get the parent module.
            module_spec = importlib.util.find_spec(proc_module.__name__)
            if module_spec.parent != '':
                parent_module = importlib.import_module(module_spec.parent)

                # Get all the modules inside the parent (namespace) module.
                # This is required here, because the namespace module can span
                # multiple repositories.
                namespace_module_infos = list(
                    pkgutil.iter_modules(
                        parent_module.__path__,
                        parent_module.__name__ + "."
                    )
                )

                # Extract the directory name of each module.
                for _, name, _ in namespace_module_infos:
                    module = importlib.import_module(name)
                    module_dir_name = os.path.dirname(inspect.getfile(module))
                    dir_names.append(module_dir_name)

        # Go through all directories and extract all the ProcModels.
        for dir_name in dir_names:
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
                            pm = ProcGroupDiGraphs._find_proc_models_in_module(
                                proc, module)
                            for proc_model in pm:
                                proc_cls_mod = \
                                    inspect.getmodule(proc).__package__ + \
                                    '.' + proc_model.__module__
                                proc_cls_mod = importlib. \
                                    import_module(proc_cls_mod)
                                class_ = getattr(proc_cls_mod,
                                                 proc_model.__name__)
                                if class_ not in proc_models:
                                    proc_models.append(class_)
                    except Exception:
                        warnings.warn(
                            f"Cannot import module '{module}' when searching "
                            f"ProcessModels for Process "
                            f"'{proc.__class__.__name__}'."
                        )

        if not proc_models:
            raise ex.NoProcessModelFound(proc)
        return proc_models

    @staticmethod
    def _select_proc_models(
            proc: AbstractProcess,
            models: ty.List[ty.Type[AbstractProcessModel]],
            run_cfg: RunConfig) -> ty.Type[AbstractProcessModel]:
        """Select a ProcessModel from list of provided models given RunCfg.

        Parameters
        ----------
        proc : AbstractProcess
            The Process for which a ProcessModel needs to be selected.
        models : List[AbstractProcessModel]
            A list of ProcessModels that implement a behaviour for `proc`.
        run_cfg : RunConfig
            The run configuration that specifies how the rules for selecting
            a ProcessModel for `proc`.

        Returns
        -------
        selected_proc_model : AbstractProcessModel
            The ProcessModel for `proc`, selected according to the rules
            specified in `run_cfg`.
        """
        selected_proc_model = run_cfg.select(proc, models)
        err_msg = f"RunConfig {run_cfg.__class__.__qualname__}.select() must " \
                  f"return a sub-class of AbstractProcessModel. Got" \
                  f" {type(selected_proc_model)} instead."
        if not isinstance(selected_proc_model, type):
            raise AssertionError(err_msg)
        if not issubclass(selected_proc_model, AbstractProcessModel):
            raise AssertionError(err_msg)

        return selected_proc_model

    @staticmethod
    def _propagate_var_ports(proc: AbstractProcess):
        """Propagate VarPorts to the SubProcesses of a hierarchical Process,
        through its SubProcessModel.

        First check if the hierarchical process has any VarPorts configured
        and if they reference an aliased Var. If so, then an implicit VarPort
        is created in the sub process and the VarPort is connected to the
        new implicit VarPort.

        This is needed since a RefPort or VarPort of a sub process cannot be
        directly targeted from processes of a different parent process.

        Parameters
        ----------
        proc : AbstractProcess
            The Process (hierarchical) whose VarPorts need to be propagated
            to its SubProcesses
        """

        for vp in proc.var_ports:
            v = vp.var.aliased_var
            if v is not None:
                # Create an implicit Var port in the sub process
                imp_vp = RefPort.create_implicit_var_port(v)
                # Connect the VarPort to the new VarPort
                vp.connect(imp_vp)

    @staticmethod
    def _expand_sub_proc_model(model_cls: ty.Type[AbstractSubProcessModel],
                               proc: AbstractProcess, run_cfg: RunConfig,
                               elab_procs : ty.List[AbstractProcess]):
        """Expand a SubProcessModel by building it, extracting the
        sub-Processes contained within, and mapping the sub-Processes
        recursively to their ProcessModels.

        Parameters
        ----------
        model_cls : AbstractSubProcessModel
            The SubProcessModel that needs expansion.
        proc : AbstractProcess
            The Process of which `model_cls` is the SubProcessModel.
        run_cfg : RunConfig
            Run configuration with which Process.run() was called.

        Returns
        -------
        A dictionary with sub-Processes as keys and their ProcessModels as
        values.
        """

        # Build SubProcessModel
        model = model_cls(proc)
        # Check if VarPorts are configured and propagate them to the sub process
        ProcGroupDiGraphs._propagate_var_ports(proc)
        # Discover sub processes and register with parent process
        sub_procs = model.find_sub_procs()
        proc.register_sub_procs(sub_procs)
        proc.validate_var_aliases()
        # Recursively map sub processes to their ProcModel
        return ProcGroupDiGraphs._map_proc_to_model(list(sub_procs.values()),
                                                    run_cfg,
                                                    elab_procs)

    @staticmethod
    def _map_proc_to_model(procs: ty.List[AbstractProcess],
                           run_cfg: RunConfig,
                           elab_procs : ty.List[AbstractProcess]) -> ProcMap:
        """Associate each Process in a list of Processes to the corresponding
        ProcessModel as selected by run_cfg.

        If a hierarchical process selected, it is expanded recursively.

        Parameters
        ----------
        procs : List[AbstractProcess]
            A list of Processes, which need to be associated with their
            ProcessModels.
        run_cfg : RunConfig
            A run configuration, according to which the ProcessModels
            for the Processes in `procs` are selected.

        Returns
        -------
        proc_map : Dict[AbstractProcess, AbstractProcessModel]
            A dictionary with Processes as keys and their corresponding
            ProcessModels as values.
        """

        proc_map = OrderedDict()
        for proc in procs:
            # Select a specific ProcessModel
            if hasattr(run_cfg, "exception_proc_model_map") and \
                    proc in run_cfg.exception_proc_model_map:
                model_cls = run_cfg.exception_proc_model_map[proc]
            else:
                models_cls = ProcGroupDiGraphs._find_proc_models(proc=proc)
                model_cls = ProcGroupDiGraphs._select_proc_models(proc,
                                                                  models_cls,
                                                                  run_cfg)
            if model_cls and issubclass(model_cls, AbstractSubProcessModel):
                # Recursively substitute SubProcModel by sub processes
                sub_map = ProcGroupDiGraphs._expand_sub_proc_model(model_cls,
                                                                   proc,
                                                                   run_cfg,
                                                                   elab_procs)
                proc_map.update(sub_map)
                proc._model_class = model_cls
            elif model_cls:
                # Just map current Process to selected ProcessModel
                proc_map[proc] = model_cls

            elab_procs.append(proc)

        return proc_map

    def _collapse_isomodel_procs(self) -> ProcDiGraph:
        """Find connected sub-graphs of Processes with same type of
        ProcessModels and collapse them onto a single Process,
        while conserving connectivity.

        Returns
        -------
        rpg : ProcDiGraph
            A directed graph derived from `resolved_proc_digraph`,
            with iso-model Process nodes collapsed onto a single node.
        """
        # Create a working copy of the resolved_proc_graph
        rpg = self._resolved_proc_digraph.copy()
        # 1. Create subgraphs of Processes with the same type of model
        subgraph_list = []
        for pm_type in ProcessModelTypes:
            subgraph_nodes = [node for node in rpg.nodes if issubclass(
                node.model_class, pm_type.value)]
            subgraph_list.append(rpg.subgraph(subgraph_nodes).copy())
        for model_sg in subgraph_list:
            # Subgraphs can be disconnected. Iterate over each weakly
            # connected component
            for wcc in ntx.weakly_connected_components(model_sg):
                if len(wcc) <= 1:
                    continue
                wccsg = model_sg.subgraph(wcc).copy()
                rpg = rpg.collapse_subgraph_to_node(wccsg)

        return rpg

    def _flatten_node_of_proc_group_digraph(self, node):
        """Flatten a node of LavaProcGraph.proc_group_digraph.

        The nodes of LavaProcGraph.proc_group_digraph can be of type
        AbstractProcess or NetworkX.DiGraph. If they are latter, then they
        are recursively flattened into a list of AbstractProcess objects.

        Parameters
        ----------
        node : ty.Any
            Either an AbstractProcess or a networkx.DiGraph. In the case of
            the latter, a list of AbstractProcesses is recursively generated
            by iterating over its nodes.

        Returns
        -------
        proc_list : List[AbstractProcess]
            A flattened list of AbstractProcesses that make up `node`.
        """
        proc_list = []
        if isinstance(node, AbstractProcess):
            proc_list.append(node)

        if isinstance(node, ntx.DiGraph):
            proc_list.extend([self._flatten_node_of_proc_group_digraph(n) for n
                              in node.nodes])
            proc_list = flatten_list_recursive(proc_list)

        return proc_list

    def get_proc_groups(self) -> \
            ty.List[ProcGroup]:
        """
        Create a list of process groups sorted in compilation order.

        The compilation order is obtained by reversing the topologically sorted
        order on compile_group_graph. Nodes of compile_group_graph can be
        graphs themselves. These are flattened into list of processes
        comprising a ProcGroup.

        This is the interface to the downstream parts of the compiler.

        Returns
        -------
        proc_groups : List[ProcGroup]
            A list of ProcGroup. The object ProcGroup is itself a short-hand
            for List[AbstractProcess]. Each ProcGroup is processed as a
            single entity by the downstream parts of the compiler.
        """

        proc_groups = []

        topo_ord = list(reversed(list(ntx.topological_sort(
            self._proc_group_digraph))))

        for node in topo_ord:
            proc_groups.append(self._flatten_node_of_proc_group_digraph(node))

        return proc_groups
