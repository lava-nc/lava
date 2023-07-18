# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import typing as ty
import importlib.util
import enum

from lava.utils import slurm


host: ty.Optional[str] = None


class ChipGeneration(enum.Enum):
    """ ChipGeneration enumerates the valid Loihi chip generations. """
    N2A2 = "N2A2"
    N3B2 = "N3B2"
    N3B3 = "N3B3"  # Most Loihi 2 systems available to the INRC are N3B3.
    N3C1 = "N3C1"  # Some Loihi 2 systems provided to INRC members are N3C1.
    N3D1 = "N3D1"


def use_slurm_host(
        partition: ty.Optional[str] = None,
        board: ty.Optional[str] = None,
        loihi_gen: ty.Optional[ChipGeneration] = ChipGeneration.N3B3
) -> None:
    """ Use SLURM to run Lava models on Loihi 2. This function should be
    called prior to running models on the Intel neuromorphic research cloud,
    or if you have setup a SLURM scheduler on your local infrastructure.

    This function checks whether Lava-Loihi is installed and raises an
    ImportError if it is not found.

    Parameters
    ----------
    partition : Optional[str], default = None
        The SLURM partition from which a suitable node should be selected. If
        partition is specified, board should be None.
    board : Optional[str], default = None
        The SLURM board (node name) on which any Lava process should run. If
        board is specified, partition should be None.
    loihi_gen : Optional[str], default = ChipGeneration.N3B3
        The Loihi chip generation needed for the Lava processes.
    """
    if not is_installed():
        raise ImportError("Attempting to use SLURM for Loihi2 but "
                          "Lava-Loihi is not installed.")

    slurm.enable()

    os.environ["LOIHI_GEN"] = loihi_gen.value

    slurm.set_board(board, partition)
    slurm.set_partition(partition)

    global host
    host = "SLURM"


def use_ethernet_host(
        host_address: str,
        host_binary_path: ty.Optional[str] = "nxcore/bin/nx_driver_server",
        loihi_gen: ty.Optional[ChipGeneration] = ChipGeneration.N3B3
) -> None:
    """Set environment to connect directly to an Oheo Gulch host on the network.
    This should be used to run on Kapoho Point and Kapoho Point SC systems when
    not using SLURM.

    This function checks whether Lava-Loihi is installed and raises an
    ImportError if it is not found.

    This function attempts to ping the host address to ensure that the
    host is running and accessible. If ping fails, it raises a ValueError.

    Call slurm.is_available() to determine whether SLURM is available.

    Parameters
    ----------
    host_address : str
        The IP address of the host system to use.
    host_binary_path : str
        The path to the nxcore binary on the host.
    loihi_gen : ChipGeneration
        The generation of the Loihi board to compile. Supported
        values are N3B2, N3B3, and N3C1.
    """
    if not is_installed():
        raise ImportError("Attempting to use Loihi2 but Lava-Loihi is "
                          "not installed.")

    if not slurm.try_run_command(["ping", host_address, "-c 1"]):
        raise ValueError(f"Attempting to use ethernet host for Loihi2 "
                         f"but `ping {host_address}` failed.")

    slurm.disable()

    os.environ["NXSDKHOST"] = host_address
    os.environ["HOST_BINARY"] = host_binary_path
    os.environ["LOIHI_GEN"] = loihi_gen.value

    global host
    host = "ETHERNET"


def is_installed(module_name: str = "lava.utils.loihi2_profiler") -> bool:
    """Returns whether the Lava extension for Loihi is installed.

    Parameters
    ----------
    module_name : Optional[str]
        Name of the module to check for checking install.

    Returns
    -------
    bool
        True iff lava-loihi can be imported in this Python environment.
    """
    spec = importlib.util.find_spec(module_name)

    return False if spec is None else True
