# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import HeadNode, Loihi1System, Loihi2System
from lava.magma.compiler.node import Node, NodeConfig


class MockProcess(AbstractProcess):
    """A mock process"""


class TestNode(unittest.TestCase):
    def test_node_creation(self):
        """Tests Node creation"""
        node1: Node = Node(node_type=HeadNode, processes=[])
        self.assertEqual(node1.node_type, HeadNode)
        self.assertEqual(len(node1.processes), 0)
        node2: Node = Node(
            node_type=Loihi1System, processes=[MockProcess(), MockProcess()]
        )
        self.assertEqual(node2.node_type, Loihi1System)
        self.assertEqual(len(node2.processes), 2)
        self.assertTrue(
            all([isinstance(p, AbstractProcess) for p in node2.processes])
        )

    def test_node_config_creation(self):
        """Tests Node Config class which is a collection of nodes"""
        node_config: NodeConfig = NodeConfig()
        loop_counter: int = 3
        compute_resource_types = [HeadNode, Loihi1System, Loihi2System]
        for i in range(loop_counter):
            node: Node = Node(node_type=compute_resource_types[i], processes=[])
            node_config.append(node)

        assert len(node_config) == loop_counter
        for i in range(loop_counter):
            assert node_config[i].node_type == compute_resource_types[i]

        node_id_list = []
        for i, node in enumerate(node_config):
            assert node.id not in node_id_list
            node_id_list.append(node.id)
            assert node.node_type == compute_resource_types[i]

        assert len(node_config) == loop_counter

        n: Node = Node(HeadNode, [])
        node_config_with_list: NodeConfig = NodeConfig([n, n, n, n])
        assert len(node_config_with_list) == 4


if __name__ == "__main__":
    unittest.main()
