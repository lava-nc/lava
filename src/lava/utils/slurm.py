# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from __future__ import annotations
import os
import subprocess  # nosec - commands are trusted
import typing as ty
from dataclasses import dataclass


def is_available() -> bool:
    """Returns true iff the current system has a SLURM controller enabled."""
    if not try_run_command(["sinfo"]):
        return False
    return True


def enable() -> None:
    if not is_available():
        raise ValueError("Attempting to use SLURM for Loihi2 but "
                         "SLURM controller is not available.")

    os.environ["SLURM"] = "1"
    os.environ.pop("NOSLURM", None)


def disable() -> None:
    os.environ.pop("SLURM", None)
    os.environ["NOSLURM"] = "1"


def set_board(board: str,
              partition: ty.Optional[str] = None) -> None:
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


def partition() -> str:
    """Get the partition information."""
    if "PARTITION" in os.environ.keys():
        return os.environ["PARTITION"]

    return "Unspecified"


def get_partitions() -> ty.List[PartitionInfo]:
    """Returns the list of available partitions from the SLURM controller
    or an empty list if SLURM is not available or has no partitions.

    Returns
    -------
    List[PartitionInfo]
        A list of all available partitions.
    """
    if not is_available():
        return []

    lines = try_run_command(["sinfo"])
    del lines[0]  # Remove header of table

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
        The partition information for the partition or None if the SLURM
        controller does not have the specified partition.
    """
    matching_partitions = [p for p in get_partitions()
                           if p.name == partition_name]

    return next(iter(matching_partitions), None)


@dataclass
class PartitionInfo:
    name: str = ""
    available: str = ""
    timelimit: str = ""
    nodes: str = ""
    state: str = ""
    nodelist: str = ""


def get_boards() -> ty.List[BoardInfo]:
    """Returns the list of available boards from the SLURM controller
    or an empty list if SLURM is not available or has no boards.

    Returns
    -------
    List[BoardInfo]
        A list of all available boards.
    """
    if not is_available():
        return []

    lines = try_run_command(["sinfo", "-N"])
    del lines[0]  # Remove header of table

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
    nodename: str = ""
    partition: str = ""
    state: str = ""


def try_run_command(command: ty.List[str]) -> ty.List[str]:
    """Executes a command, captures the output, and splits it into a list of
    lines (strings). Returns an empty list if executing the command raises
    and exception.

    Parameters
    ----------
    command : List[str]
        Command and options, for instance 'sinfo -N' becomes ['sinfo', '-N']

    Returns
    -------
    List[str]
        Output of stdout of the command, separated into a list of lines (str).
    """
    try:
        kwargs = dict(capture_output=True, check=True, timeout=1)
        process = subprocess.run(command, text=True, **kwargs)  # nosec # noqa
        return process.stdout.split("\n")

    except subprocess.SubprocessError:
        return []
