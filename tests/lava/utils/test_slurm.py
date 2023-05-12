# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
from unittest.mock import patch
import typing as ty
import os

from lava.utils import slurm


class TestSlurm(unittest.TestCase):
    def sinfo(self) -> ty.List[str]:
        """Simulates a shell call to 'sinfo'."""
        sinfo_msg = \
            """PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
            partition1   up    30:00     1     idle  node1
            partition2   up    30:00     1     down  node2"""
        return sinfo_msg.split("\n")

    def sinfo_n(self) -> ty.List[str]:
        """Simulates a shell call to 'sinfo -N'."""
        sinfo_msg = \
            """NODELIST NODES PARTITION  STATE
            node1       1     partition1 idle
            node2       1     partition2 down"""
        return sinfo_msg.split("\n")

    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_boards_when_slurm_is_available(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = self.sinfo_n()

        boards = slurm.get_boards()

        self.assertEqual(len(boards), 2)

        self.assertEqual(boards[0].nodename, "node1")
        self.assertEqual(boards[0].partition, "partition1")
        self.assertEqual(boards[0].state, "idle")

        self.assertEqual(boards[1].nodename, "node2")
        self.assertEqual(boards[1].partition, "partition2")
        self.assertEqual(boards[1].state, "down")

    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_boards_when_slurm_is_not_available(
            self, is_available, try_run_command) -> None:
        is_available.return_value = False
        try_run_command.return_value = self.sinfo_n()

        boards = slurm.get_boards()

        self.assertEqual(boards, [])

    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_board_info_for_existing_board(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = self.sinfo_n()

        board_info = slurm.get_board_info("node1")

        self.assertEqual(board_info.nodename, "node1")
        self.assertEqual(board_info.partition, "partition1")
        self.assertEqual(board_info.state, "idle")

    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_board_info_for_non_existing_board(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = self.sinfo_n()

        board_info = slurm.get_board_info("non_existing_node")

        self.assertEqual(board_info, None)

    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_partitions_when_slurm_is_available(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = self.sinfo()

        partitions = slurm.get_partitions()

        self.assertEqual(len(partitions), 2)

        self.assertEqual(partitions[0].name, "partition1")
        self.assertEqual(partitions[0].available, "up")
        self.assertEqual(partitions[0].timelimit, "30:00")
        self.assertEqual(partitions[0].nodes, "1")
        self.assertEqual(partitions[0].state, "idle")
        self.assertEqual(partitions[0].nodelist, "node1")

        self.assertEqual(partitions[1].name, "partition2")
        self.assertEqual(partitions[1].available, "up")
        self.assertEqual(partitions[1].timelimit, "30:00")
        self.assertEqual(partitions[1].nodes, "1")
        self.assertEqual(partitions[1].state, "down")
        self.assertEqual(partitions[1].nodelist, "node2")

    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_partitions_when_no_boards_are_available(
            self, is_available, try_run_command) -> None:
        is_available.return_value = False
        try_run_command.return_value = self.sinfo()

        partitions = slurm.get_partitions()

        self.assertEqual(partitions, [])

    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_partition_info_for_existing_partition(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = self.sinfo()

        partition_info = slurm.get_partition_info("partition1")

        self.assertEqual(partition_info.name, "partition1")
        self.assertEqual(partition_info.available, "up")
        self.assertEqual(partition_info.timelimit, "30:00")
        self.assertEqual(partition_info.nodes, "1")
        self.assertEqual(partition_info.state, "idle")
        self.assertEqual(partition_info.nodelist, "node1")

    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_partition_info_for_non_existing_partition(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = self.sinfo()

        board_info = slurm.get_partition_info("non_existing_partition")

        self.assertEqual(board_info, None)

    @patch("lava.utils.slurm.try_run_command")
    def test_is_available_true(self, try_run_command) -> None:
        try_run_command.return_value = self.sinfo()

        self.assertTrue(slurm.is_available())

    @patch("lava.utils.slurm.try_run_command")
    def test_is_available_false(self, try_run_command) -> None:
        try_run_command.return_value = []

        self.assertFalse(slurm.is_available())

    @patch.dict(os.environ, {"PARTITION": "test_partition"}, clear=True)
    def test_partition_available(self) -> None:
        partition = slurm.partition()

        self.assertEqual(partition, "test_partition")

    @patch.dict(os.environ, {"VARIABLE": "VALUE"}, clear=True)
    def test_partition_not_available(self) -> None:
        partition = slurm.partition()

        self.assertEqual(partition, "Unspecified")
