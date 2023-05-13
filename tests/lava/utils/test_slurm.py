# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import subprocess
import unittest
from unittest.mock import patch, MagicMock
import typing as ty
import os

from lava.utils import slurm


def sinfo() -> ty.List[str]:
    """Simulates a shell call to 'sinfo'."""
    sinfo_msg = \
        """PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
        partition1   up    30:00     1     idle  node1
        partition2   up    30:00     1     down  node2"""
    return sinfo_msg.split("\n")


def sinfo_n() -> ty.List[str]:
    """Simulates a shell call to 'sinfo -N'."""
    sinfo_msg = \
        """NODELIST NODES PARTITION  STATE
        node1       1     partition1 idle
        node2       1     partition2 down"""
    return sinfo_msg.split("\n")


class TestUseSlurmHost(unittest.TestCase):
    @patch.dict(os.environ, {"NOSLURM": "1"}, clear=True)
    @patch("lava.utils.slurm.get_board_info")
    @patch("lava.utils.slurm.get_partition_info")
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.is_available")
    def test_use_slurm_host_with_board_and_partition(
            self,
            is_available,
            is_installed,
            get_partition_info,
            get_board_info) -> None:
        is_available.return_value = True
        is_installed.return_value = True
        get_partition_info.return_value = slurm.PartitionInfo()
        get_board_info.return_value = slurm.BoardInfo(partition="partition1")

        slurm.use_slurm_host(board="board1",
                             partition="partition1",
                             loihi_gen=slurm.LoihiGeneration.N3C1)

        self.assertEqual(os.environ["SLURM"], "1")
        self.assertEqual(os.environ["LOIHI_GEN"], "N3C1")
        self.assertEqual(os.environ["BOARD"], "board1")
        self.assertEqual(os.environ["PARTITION"], "partition1")
        self.assertTrue("NOSLURM" not in os.environ.keys())
        self.assertEqual(slurm.host, "SLURM")

    @patch.dict(os.environ, {"NOSLURM": "1", "PARTITION": "test"}, clear=True)
    @patch("lava.utils.slurm.get_board_info")
    @patch("lava.utils.slurm.get_partition_info")
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.is_available")
    def test_use_slurm_host_with_board(
            self,
            is_available,
            is_installed,
            get_partition_info,
            get_board_info) -> None:
        is_available.return_value = True
        is_installed.return_value = True
        get_partition_info.return_value = slurm.PartitionInfo()
        get_board_info.return_value = slurm.BoardInfo()

        slurm.use_slurm_host(board="board1",
                             loihi_gen=slurm.LoihiGeneration.N3C1)

        self.assertEqual(os.environ["SLURM"], "1")
        self.assertEqual(os.environ["LOIHI_GEN"], "N3C1")
        self.assertEqual(os.environ["BOARD"], "board1")
        self.assertTrue("PARTITION" not in os.environ.keys())
        self.assertTrue("NOSLURM" not in os.environ.keys())
        self.assertEqual(slurm.host, "SLURM")

    @patch.dict(os.environ, {"NOSLURM": "1", "PARTITION": "test"}, clear=True)
    @patch("lava.utils.slurm.get_board_info")
    @patch("lava.utils.slurm.get_partition_info")
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.is_available")
    def test_use_slurm_host_with_partition(
            self,
            is_available,
            is_installed,
            get_partition_info,
            get_board_info) -> None:
        is_available.return_value = True
        is_installed.return_value = True
        get_partition_info.return_value = slurm.PartitionInfo()
        get_board_info.return_value = slurm.BoardInfo()

        slurm.use_slurm_host(partition="partition1",
                             loihi_gen=slurm.LoihiGeneration.N3C1)

        self.assertEqual(os.environ["SLURM"], "1")
        self.assertEqual(os.environ["LOIHI_GEN"], "N3C1")
        self.assertEqual(os.environ["PARTITION"], "partition1")
        self.assertTrue("BOARD" not in os.environ.keys())
        self.assertTrue("NOSLURM" not in os.environ.keys())
        self.assertEqual(slurm.host, "SLURM")

    @patch.dict(os.environ, {}, clear=True)
    @patch("lava.utils.slurm.get_board_info")
    @patch("lava.utils.slurm.get_partition_info")
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.is_available")
    def test_use_slurm_host_when_lava_loihi_is_not_installed(
            self,
            is_available,
            is_installed,
            get_partition_info,
            get_board_info) -> None:
        is_available.return_value = True
        is_installed.return_value = False
        get_partition_info.return_value = slurm.PartitionInfo()
        get_board_info.return_value = slurm.BoardInfo()

        with self.assertRaises(ImportError):
            slurm.use_slurm_host(board="board1",
                                 partition="partition1",
                                 loihi_gen=slurm.LoihiGeneration.N3C1)

    @patch.dict(os.environ, {}, clear=True)
    @patch("lava.utils.slurm.get_board_info")
    @patch("lava.utils.slurm.get_partition_info")
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.is_available")
    def test_use_slurm_host_when_slurm_is_not_available(
            self,
            is_available,
            is_installed,
            get_partition_info,
            get_board_info) -> None:
        is_available.return_value = False
        is_installed.return_value = True
        get_partition_info.return_value = slurm.PartitionInfo()
        get_board_info.return_value = slurm.BoardInfo()

        with self.assertRaises(ValueError):
            slurm.use_slurm_host(board="board1",
                                 partition="partition1",
                                 loihi_gen=slurm.LoihiGeneration.N3C1)

    @patch.dict(os.environ, {}, clear=True)
    @patch("lava.utils.slurm.get_board_info")
    @patch("lava.utils.slurm.get_partition_info")
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.is_available")
    def test_use_slurm_host_when_board_does_not_exist(
            self,
            is_available,
            is_installed,
            get_partition_info,
            get_board_info) -> None:
        is_available.return_value = True
        is_installed.return_value = True
        get_partition_info.return_value = slurm.PartitionInfo()
        get_board_info.return_value = None

        with self.assertRaises(ValueError):
            slurm.use_slurm_host(board="board1",
                                 partition="partition1",
                                 loihi_gen=slurm.LoihiGeneration.N3C1)

    @patch.dict(os.environ, {}, clear=True)
    @patch("lava.utils.slurm.get_board_info")
    @patch("lava.utils.slurm.get_partition_info")
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.is_available")
    def test_use_slurm_host_when_board_is_down(
            self,
            is_available,
            is_installed,
            get_partition_info,
            get_board_info) -> None:
        is_available.return_value = True
        is_installed.return_value = True
        get_partition_info.return_value = slurm.PartitionInfo()
        get_board_info.return_value = slurm.BoardInfo(state="down")

        with self.assertRaises(ValueError):
            slurm.use_slurm_host(board="board1",
                                 partition="partition1",
                                 loihi_gen=slurm.LoihiGeneration.N3C1)

    @patch.dict(os.environ, {}, clear=True)
    @patch("lava.utils.slurm.get_board_info")
    @patch("lava.utils.slurm.get_partition_info")
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.is_available")
    def test_use_slurm_host_when_board_is_not_in_partition(
            self,
            is_available,
            is_installed,
            get_partition_info,
            get_board_info) -> None:
        is_available.return_value = True
        is_installed.return_value = True
        get_partition_info.return_value = slurm.PartitionInfo()
        get_board_info.return_value = slurm.BoardInfo(partition="not-p1")

        with self.assertRaises(ValueError):
            slurm.use_slurm_host(board="board1",
                                 partition="p1",
                                 loihi_gen=slurm.LoihiGeneration.N3C1)

    @patch.dict(os.environ, {}, clear=True)
    @patch("lava.utils.slurm.get_board_info")
    @patch("lava.utils.slurm.get_partition_info")
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.is_available")
    def test_use_slurm_host_when_partition_does_not_exist(
            self,
            is_available,
            is_installed,
            get_partition_info,
            get_board_info) -> None:
        is_available.return_value = True
        is_installed.return_value = True
        get_partition_info.return_value = None
        get_board_info.return_value = slurm.BoardInfo()

        with self.assertRaises(ValueError):
            slurm.use_slurm_host(board="board1",
                                 partition="partition1",
                                 loihi_gen=slurm.LoihiGeneration.N3C1)

    @patch.dict(os.environ, {}, clear=True)
    @patch("lava.utils.slurm.get_board_info")
    @patch("lava.utils.slurm.get_partition_info")
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.is_available")
    def test_use_slurm_host_when_partition_is_down(
            self,
            is_available,
            is_installed,
            get_partition_info,
            get_board_info) -> None:
        is_available.return_value = True
        is_installed.return_value = True
        get_partition_info.return_value = slurm.PartitionInfo(state="down")
        get_board_info.return_value = slurm.BoardInfo()

        with self.assertRaises(ValueError):
            slurm.use_slurm_host(board="board1",
                                 partition="partition1",
                                 loihi_gen=slurm.LoihiGeneration.N3C1)


class TestUseEthernetHost(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.try_run_command")
    def test_use_ethernet_host(
            self,
            try_run_command,
            is_installed) -> None:
        try_run_command.return_value = ["1 packets transmitted, 1 received"]
        is_installed.return_value = True

        slurm.use_ethernet_host(host_address="test_address",
                                host_binary_path="test_path",
                                loihi_gen=slurm.LoihiGeneration.N3C1)

        self.assertEqual(os.environ["NXSDKHOST"], "test_address")
        self.assertEqual(os.environ["HOST_BINARY"], "test_path")
        self.assertEqual(os.environ["NOSLURM"], "1")
        self.assertEqual(os.environ["LOIHI_GEN"], "N3C1")
        self.assertEqual(slurm.host, "ETHERNET")

    @patch("lava.utils.lava_loihi.is_installed")
    def test_use_ethernet_host_when_lava_loihi_is_not_installed(
            self,
            is_installed) -> None:
        is_installed.return_value = False

        with self.assertRaises(ImportError):
            slurm.use_ethernet_host(host_address="test_host",
                                    host_binary_path="test_path")

    @patch("lava.utils.lava_loihi.is_installed")
    @patch("lava.utils.slurm.try_run_command")
    def test_use_ethernet_host_when_ping_fails(
            self,
            try_run_command,
            is_installed) -> None:
        try_run_command.return_value = []
        is_installed.return_value = True

        with self.assertRaises(ValueError):
            slurm.use_ethernet_host(host_address="test_host",
                                    host_binary_path="test_path")


class TestPartition(unittest.TestCase):
    @patch.dict(os.environ, {"PARTITION": "test_partition"}, clear=True)
    def test_partition_available(self) -> None:
        partition = slurm.partition()

        self.assertEqual(partition, "test_partition")

    @patch.dict(os.environ, {"VARIABLE": "VALUE"}, clear=True)
    def test_partition_not_available(self) -> None:
        partition = slurm.partition()

        self.assertEqual(partition, "Unspecified")


class TestIsAvailable(unittest.TestCase):
    @patch("lava.utils.slurm.try_run_command")
    def test_is_available_true(self, try_run_command) -> None:
        try_run_command.return_value = sinfo()

        self.assertTrue(slurm.is_available())

    @patch("lava.utils.slurm.try_run_command")
    def test_is_available_false(self, try_run_command) -> None:
        try_run_command.return_value = []

        self.assertFalse(slurm.is_available())


class TestGetPartitions(unittest.TestCase):
    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_partitions_when_slurm_is_available(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = sinfo()

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
        try_run_command.return_value = sinfo()

        partitions = slurm.get_partitions()

        self.assertEqual(partitions, [])


class TestGetPartitionInfo(unittest.TestCase):
    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_partition_info_for_existing_partition(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = sinfo()

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
        try_run_command.return_value = sinfo()

        board_info = slurm.get_partition_info("non_existing_partition")

        self.assertEqual(board_info, None)


class TestGetBoards(unittest.TestCase):
    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_boards_when_slurm_is_available(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = sinfo_n()

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
        try_run_command.return_value = sinfo_n()

        boards = slurm.get_boards()

        self.assertEqual(boards, [])


class TestGetBoardInfo(unittest.TestCase):
    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_board_info_for_existing_board(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = sinfo_n()

        board_info = slurm.get_board_info("node1")

        self.assertEqual(board_info.nodename, "node1")
        self.assertEqual(board_info.partition, "partition1")
        self.assertEqual(board_info.state, "idle")

    @patch("lava.utils.slurm.try_run_command")
    @patch("lava.utils.slurm.is_available")
    def test_get_board_info_for_non_existing_board(
            self, is_available, try_run_command) -> None:
        is_available.return_value = True
        try_run_command.return_value = sinfo_n()

        board_info = slurm.get_board_info("non_existing_node")

        self.assertEqual(board_info, None)


class TestTryRunCommand(unittest.TestCase):
    @patch("subprocess.run")
    def test_output_is_split_into_lines(self, run) -> None:
        process = MagicMock(spec="subprocess.CompletedProcess")
        process.stdout = "line1\nline2"
        run.return_value = process

        output = slurm.try_run_command(["test_command"])

        self.assertEqual(output, ["line1", "line2"])

    @patch("subprocess.run")
    def test_output_returns_empty_list_on_exception(self, run) -> None:
        run.side_effect = subprocess.SubprocessError

        output = slurm.try_run_command(["test_command"])

        self.assertEqual(output, [])
