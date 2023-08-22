# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import unittest
from unittest.mock import MagicMock, patch
import typing as ty

from lava.utils import loihi, slurm


def patch_use_slurm_host(environ: ty.Dict[str, ty.Any]) -> ty.Callable:
    """Decorator to enable reuse of a set of patches.

    Parameters
    ----------
    environ : dict
        Dictionary that replaces all environment variables (os.environ) for
        the duration of the test.
    """

    def decorator(test_function: ty.Callable) -> ty.Callable:
        @patch.dict(os.environ, environ, clear=True)
        @patch("lava.utils.slurm.get_board_info")
        @patch("lava.utils.slurm.get_partition_info")
        @patch("lava.utils.loihi.is_installed")
        @patch("lava.utils.slurm.is_available")
        def wrapper(*args) -> None:
            test_function(*args)

        return wrapper

    return decorator


class TestUseSlurmHost(unittest.TestCase):
    @patch_use_slurm_host(environ={})
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

        loihi.use_slurm_host(board="board1",
                             partition="partition1",
                             loihi_gen=loihi.ChipGeneration.N3C1)

        self.assertEqual(os.environ["LOIHI_GEN"], "N3C1")
        self.assertEqual(os.environ["BOARD"], "board1")
        self.assertEqual(os.environ["PARTITION"], "partition1")
        self.assertEqual(loihi.host, "SLURM")

    @patch_use_slurm_host(environ={"PARTITION": "test"})
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

        loihi.use_slurm_host(board="board1",
                             loihi_gen=loihi.ChipGeneration.N3C1)

        self.assertEqual(os.environ["LOIHI_GEN"], "N3C1")
        self.assertEqual(os.environ["BOARD"], "board1")
        self.assertTrue("PARTITION" not in os.environ.keys())
        self.assertEqual(loihi.host, "SLURM")

    @patch_use_slurm_host(environ={"PARTITION": "test"})
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

        loihi.use_slurm_host(partition="partition1",
                             loihi_gen=loihi.ChipGeneration.N3C1)

        self.assertEqual(os.environ["LOIHI_GEN"], "N3C1")
        self.assertEqual(os.environ["PARTITION"], "partition1")
        self.assertTrue("BOARD" not in os.environ.keys())
        self.assertEqual(loihi.host, "SLURM")

    @patch_use_slurm_host(environ={})
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
            loihi.use_slurm_host(board="board1",
                                 partition="partition1",
                                 loihi_gen=loihi.ChipGeneration.N3C1)

    @patch_use_slurm_host(environ={})
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
            loihi.use_slurm_host(board="board1",
                                 partition="partition1",
                                 loihi_gen=loihi.ChipGeneration.N3C1)


class TestUseEthernetHost(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    @patch("lava.utils.loihi.is_installed")
    @patch("lava.utils.slurm.try_run_command")
    def test_use_ethernet_host(
            self,
            try_run_command,
            is_installed) -> None:
        try_run_command.return_value = ["1 packets transmitted, 1 received"]
        is_installed.return_value = True

        loihi.use_ethernet_host(host_address="test_address",
                                host_binary_path="test_path",
                                loihi_gen=loihi.ChipGeneration.N3C1)

        self.assertEqual(os.environ["NXSDKHOST"], "test_address")
        self.assertEqual(os.environ["HOST_BINARY"], "test_path")
        self.assertEqual(os.environ["LOIHI_GEN"], "N3C1")
        self.assertEqual(loihi.host, "ETHERNET")

    @patch("lava.utils.loihi.is_installed")
    def test_use_ethernet_host_when_lava_loihi_is_not_installed(
            self,
            is_installed) -> None:
        is_installed.return_value = False

        with self.assertRaises(ImportError):
            loihi.use_ethernet_host(host_address="test_host",
                                    host_binary_path="test_path")

    @patch("lava.utils.loihi.is_installed")
    @patch("lava.utils.slurm.try_run_command")
    def test_use_ethernet_host_when_ping_fails(
            self,
            try_run_command,
            is_installed) -> None:
        try_run_command.return_value = []
        is_installed.return_value = True

        with self.assertRaises(ValueError):
            loihi.use_ethernet_host(host_address="test_host",
                                    host_binary_path="test_path")


class TestIsInstalled(unittest.TestCase):
    def test_on_known_installed_lava_module(self) -> None:
        """Tests whether the method for checking on an installed library
        works in general."""
        self.assertTrue(loihi.is_installed("lava"))

    @patch("importlib.util.find_spec")
    def test_is_installed(self, find_spec) -> None:
        """Tests whether the function detects when Lava-Loihi is installed."""
        find_spec.return_value = \
            MagicMock(spec="importlib.machinery.ModuleSpec")

        self.assertTrue(loihi.is_installed())

    @patch("importlib.util.find_spec")
    def test_is_not_installed(self, find_spec) -> None:
        """Tests whether the function detects when Lava-Loihi is not
        installed."""
        find_spec.return_value = None

        self.assertFalse(loihi.is_installed())
