# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import unittest
from unittest.mock import Mock

from lava.magma.compiler.executable import Executable
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.core.resources import HeadNode, Loihi2System
from lava.magma.compiler.node import Node, NodeConfig
from lava.magma.compiler.channels.watchdog import WatchdogManagerBuilder
from lava.magma.runtime.runtime import Runtime


class TestRuntime(unittest.TestCase):
    def test_runtime_creation(self):
        """Tests runtime constructor"""
        exe = Mock(spec_set=Executable)
        mp = ActorType.MultiProcessing
        runtime: Runtime = Runtime(exe=exe,
                                   message_infrastructure_type=mp)
        expected_type: ty.Type = Runtime
        self.assertIsInstance(
            runtime, expected_type,
            f"Expected type {expected_type} doesn't match {(type(runtime))}")

    def test_executable_node_config_assertion(self):
        """Tests runtime constructions with expected constraints"""
        compile_config = {"long_event_timeout": 10,
                          "short_event_timeout": 10,
                          "use_watchdog": False}
        w = WatchdogManagerBuilder(compile_config, 30)
        exe: Executable = Executable(process_list=[],
                                     proc_builders={},
                                     channel_builders=[],
                                     node_configs=[],
                                     sync_domains=[],
                                     watchdog_manager_builder=w
                                     )

        runtime1: Runtime = Runtime(exe, ActorType.MultiProcessing)
        runtime1.initialize()

        node: Node = Node(HeadNode, [])
        exe.node_configs.append(NodeConfig([node]))
        runtime2: Runtime = Runtime(exe, ActorType.MultiProcessing)
        runtime2.initialize()
        expected_type: ty.Type = Runtime
        self.assertIsInstance(
            runtime2, expected_type,
            f"Expected type {expected_type} doesn't match {(type(runtime2))}")
        runtime2.stop()

        exe1: Executable = Executable(process_list=[],
                                      proc_builders={},
                                      channel_builders=[],
                                      node_configs=[],
                                      sync_domains=[],
                                      watchdog_manager_builder=w)
        node1: Node = Node(Loihi2System, [])
        exe1.node_configs.append(NodeConfig([node1]))
        runtime3: Runtime = Runtime(exe1, ActorType.MultiProcessing)
        runtime3.initialize(0)

        exe.node_configs.append(NodeConfig([node]))
        runtime4: Runtime = Runtime(exe, ActorType.MultiProcessing)
        runtime4.initialize(0)
        self.assertEqual(len(runtime4._executable.node_configs), 2,
                         "Expected node_configs length to be 2")
        node2: Node = Node(Loihi2System, [])
        exe.node_configs[0].append(node2)
        self.assertEqual(len(runtime4._executable.node_configs[0]), 2,
                         "Expected node_configs[0] node_config length to be 2")
        runtime4.stop()


if __name__ == "__main__":
    unittest.main()
