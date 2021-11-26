# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

from __future__ import annotations

import typing
import typing as ty
from collections import UserList, OrderedDict

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import AbstractNode
from lava.magma.compiler.exec_var import AbstractExecVar


class Node:
    """A Node represents a physical compute node on which one or more
    processes execute.

    Nodes are of a specific type and hold references to all processes mapped
    to a node."""

    def __init__(
            self,
            node_type: ty.Type[AbstractNode],
            processes: ty.List[AbstractProcess]):
        self.id: int = -1
        self.node_type: typing.Type[AbstractNode] = node_type
        self.processes = processes


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
        self.exec_vars: ty.Dict[int, AbstractExecVar] = OrderedDict()

    def append(self, node: Node):
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

    def set_exec_vars(self, exec_vars: ty.Dict[int, AbstractExecVar]):
        self.exec_vars = exec_vars
