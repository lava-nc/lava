# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

from __future__ import annotations

import typing as ty
from collections import UserList, OrderedDict

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import AbstractNode
from lava.magma.compiler.var_model import AbstractVarModel


class Node:
    """A Node represents a physical compute node on which one or more
    processes execute.

    Nodes are of a specific type and hold references to all processes mapped
    to a node."""

    def __init__(
        self,
        node_type: ty.Type[AbstractNode],
        processes: ty.List[AbstractProcess],
    ):
        self.id: int = -1
        self.node_type: ty.Type[AbstractNode] = node_type
        self.processes = processes

    def add_process(self, process: AbstractProcess):
        self.processes.append(process)

    def __str__(self):
        return f"{self.id=} {self.node_type=} {self.processes=}"


class NodeConfig(UserList):
    """A NodeConfig is a collection of Nodes. Nodes represent a physical
    compute node on which one or more processes execute.

    A NodeCfg has a list of all 'nodes' and a 'node_map' that maps each
    process to its respective node.
    """

    def __init__(self, init_list=None):
        super().__init__(init_list)
        self._node_ctr = 0
        self.node_map: ty.Dict[AbstractProcess, Node] = OrderedDict()
        self.var_models: ty.Dict[int, AbstractVarModel] = OrderedDict()

    def __str__(self):
        result = []
        result.append(f"{self._node_ctr=}")
        result.append(str(self.node_map))
        return "\n".join(result)

    def append(self, node: Node):  # pylint: disable=arguments-renamed
        """Appends a new node to the NodeConfig."""
        node.id = self._node_ctr
        self._node_ctr += 1
        super().append(node)
        for p in node.processes:
            self.node_map[p] = node

    @property
    def nodes(self) -> ty.List[Node]:
        """Returns list of all nodes of the NodeConfig."""
        return self.data

    def set_var_models(self, var_models: ty.Dict[int, AbstractVarModel]):
        self.var_models = var_models
