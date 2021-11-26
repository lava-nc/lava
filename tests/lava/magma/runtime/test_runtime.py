# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

import typing as ty
import unittest

from lava.magma.compiler.executable import Executable
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.core.resources import HeadNode
from lava.magma.core.run_conditions import RunSteps, AbstractRunCondition
from lava.magma.compiler.node import Node, NodeConfig
from lava.magma.runtime.runtime import Runtime


class TestRuntime(unittest.TestCase):
    def test_runtime_creation(self):
        """Tests runtime constructor"""
        exe: Executable = Executable()
        run_cond: AbstractRunCondition = RunSteps(num_steps=10)
        mp = ActorType.MultiProcessing
        runtime: Runtime = Runtime(run_cond=run_cond,
                                   exe=exe,
                                   message_infrastructure_type=mp)
        expected_type: ty.Type = Runtime
        assert isinstance(
            runtime, expected_type
        ), f"Expected type {expected_type} doesn't match {(type(runtime))}"

    def test_executable_node_config_assertion(self):
        """Tests runtime constructions with expected constraints"""
        exec: Executable = Executable()
        run_cond: AbstractRunCondition = RunSteps(num_steps=10)

        runtime1: Runtime = Runtime(run_cond, exec, ActorType.MultiProcessing)
        with self.assertRaises(AssertionError):
            runtime1.initialize()

        node: Node = Node(HeadNode, [])
        exec.node_configs.append(NodeConfig([node]))
        runtime2: Runtime = Runtime(run_cond, exec, ActorType.MultiProcessing)
        runtime2.initialize()
        expected_type: ty.Type = Runtime
        assert isinstance(
            runtime2, expected_type
        ), f"Expected type {expected_type} doesn't match {(type(runtime2))}"
        runtime2.stop()

        exec.node_configs[0].append(node)
        runtime3: Runtime = Runtime(run_cond, exec, ActorType.MultiProcessing)
        with self.assertRaises(AssertionError):
            runtime3.initialize()

        exec.node_configs.append(NodeConfig([node]))
        runtime4: Runtime = Runtime(run_cond, exec, ActorType.MultiProcessing)
        with self.assertRaises(AssertionError):
            runtime4.initialize()


if __name__ == "__main__":
    unittest.main()
