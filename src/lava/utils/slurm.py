# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from __future__ import annotations
import os
import subprocess  # nosec - commands are trusted
import typing as ty
from dataclasses import dataclass
import enum

from lava.utils import lava_loihi


host: ty.Optional[str] = None


class LoihiGeneration(enum.StrEnum):
    N3B2 = "N3B2"
    N3B3 = "N3B3"
    N3C1 = "N3C1"


def use_slurm_host(
        partition: ty.Optional[str] = None,
        board: ty.Optional[str] = None,
        loihi_gen: ty.Optional[LoihiGeneration] = LoihiGeneration.N3B3
) -> None:

    if not lava_loihi.is_installed():
        raise ImportError("Attempting to use SLURM for Loihi2 but "
                          "Lava-Loihi is not installed.")

    if not is_available():
        raise ValueError("Attempting to use SLURM for Loihi2 but "
                         "SLURM controller is not available.")

    os.environ["SLURM"] = "1"
    os.environ.pop("NOSLURM", None)
    os.environ["LOIHI_GEN"] = loihi_gen.value

    if board:
        set_board(board)

    if partition:
        set_partition(partition)

    global host
    host = "SLURM"


def set_board(board: str) -> None:
    board_info = get_board_info(board)

    if board_info is None or "down" in board_info.state:
        raise ValueError(
            f"Attempting to use SLURM for Loihi2 but board {board} "
            f"is not found or board is down. Run sinfo to check "
            f"available boards.")

    if partition and partition != board_info.partition:
        raise ValueError(
            f"Attempting to use SLURM for Loihi2 with board {board} "
            f"and partition {partition} but board is not in partition. "
            f"Specify only board or partition.")

    os.environ["BOARD"] = board


def set_partition(partition: str) -> None:
    partition_info = get_partition_info(partition)

    if partition_info is None or "down" in partition_info.state:
        raise ValueError(
            f"Attempting to use SLURM for Loihi2 but partition {partition} "
            f"is not found or board is down. Run sinfo to check available "
            f"boards.")

    os.environ["PARTITION"] = partition


def use_ethernet_host(
        host_address: str,
        host_binary_path: str = 'nxcore/bin/nx_driver_server',
        loihi_gen: LoihiGeneration = LoihiGeneration.N3B3) -> None:
    """Set environment to connect directly to an Oheo Gulch host on the network.
    This should be used to run on Kapoho Point and Kapoho Point SC systems when
    SLURM is not available.

    Call slurm.is_available() to determine whether SLURM is available.

    Parameters
    ----------
    host_address : str
        The IP address for the host system to connect to.
    host_binary_path : str
        The path to the nxcore binary on the host.
    loihi_gen : LoihiGeneration
        The generation of the Loihi board to compile. Supported
        values are N3B2, N3B3, and N3C1.
    """
    if not lava_loihi.is_installed():
        raise ImportError("Attempting to use SLURM for Loihi2 but "
                          "Lava-Loihi is not installed.")

    if not try_run_command(["ping", host_address, "-c 1"]):
        raise ValueError(f"Attempting to use ethernet host for Loihi2 "
                         f"but `ping {host_address}` failed.")

    os.environ["NXSDKHOST"] = host_address
    os.environ["HOST_BINARY"] = host_binary_path
    os.environ.pop("SLURM", None)

    os.environ["NOSLURM"] = "1"
    os.environ["LOIHI_GEN"] = loihi_gen.value

    global host
    host = "ETHERNET"


def partition() -> str:
    """Get the partition information."""
    if "PARTITION" in os.environ.keys():
        return os.environ["PARTITION"]

    return "Unspecified"


def is_available():
    """Returns true iff the current system has a SLURM controller enabled."""
    if try_run_command(["sinfo"]) == "":
        return False
    return True


def get_partitions() -> ty.List[PartitionInfo]:
    """Returns the list of available partitions from the SLURM controller
    or an empty list if SLURM is not available or has no partitions."""
    if not is_available():
        return []

    out = try_run_command(["sinfo"])
    lines = out.stdout.split("\n")

    def parse_partition(line: str) -> PartitionInfo:
        fields = line.split()

        return PartitionInfo(name=fields[0],
                             available=fields[1],
                             timelimit=fields[2],
                             nodes=fields[3],
                             state=fields[4],
                             nodelist=fields[5])

    return [parse_partition(line) for line in lines]


def get_partition_info(partition_name: str) -> ty.Optional[PartitionInfo]:
    """Get the SLURM info for the specified partition, if available.

    Parameters
    ----------
    partition_name : str
        The name of the partition to return.

    Returns
    -------
    Optional[PartitionInfo]
        The partition information  for the partition or None if the SLURM
        controller does not have the specified partition.
    """
    matching_partitions = [p for p in get_partitions()
                           if p.name == partition_name]

    return next(iter(matching_partitions), None)


@dataclass
class PartitionInfo:
    name: str
    available: str
    timelimit: str
    nodes: str
    state: str
    nodelist: str


def get_boards() -> ty.List[BoardInfo]:
    """Returns the list of available boards from the SLURM controller
    or an empty list if SLURM is not available or has no boards."""
    if not is_available():
        return []

    out = try_run_command(["sinfo", "-N"])
    lines = out.stdout.split("\n")

    def parse_board(line: str) -> BoardInfo:
        fields = line.split()

        return BoardInfo(nodename=fields[0],
                         partition=fields[2],
                         state=fields[3])

    return [parse_board(line) for line in lines]


def get_board_info(nodename: str) -> ty.Optional[BoardInfo]:
    """Get the SLURM info for the specified board, if available.

    Parameters
    ----------
    nodename : str
        The name of the board to return.

    Returns
    -------
    Optional[BoardInfo]
        The information for the board or None if the SLURM
        controller does not have the specified board.
    """
    matching_boards = [b for b in get_boards()
                       if b.nodename == nodename]

    return next(iter(matching_boards), None)


@dataclass
class BoardInfo:
    nodename: str
    partition: str
    state: str


def try_run_command(
        command: ty.List[str]) -> ty.Union[subprocess.CompletedProcess, str]:
    try:
        return subprocess.run(command,
                              capture_output=True,
                              text=True,
                              check=True,
                              timeout=1)  # nosec S603 - commands are trusted
    except subprocess.SubprocessError:
        return ""
