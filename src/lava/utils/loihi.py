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
    N3B2 = "N3B2"
    N3B3 = "N3B3"
    N3C1 = "N3C1"


def use_slurm_host(
        partition: ty.Optional[str] = None,
        board: ty.Optional[str] = None,
        loihi_gen: ty.Optional[ChipGeneration] = ChipGeneration.N3B3
) -> None:
    if not is_lava_loihi_installed():
        raise ImportError("Attempting to use SLURM for Loihi2 but "
                          "Lava-Loihi is not installed.")

    slurm.enable()

    os.environ["LOIHI_GEN"] = loihi_gen.value

    if board:
        slurm.set_board(board, partition)
    else:
        os.environ.pop("BOARD", None)

    if partition:
        slurm.set_partition(partition)
    else:
        os.environ.pop("PARTITION", None)

    global host
    host = "SLURM"


def use_ethernet_host(
        host_address: str,
        host_binary_path: ty.Optional[str] = "nxcore/bin/nx_driver_server",
        loihi_gen: ty.Optional[ChipGeneration] = ChipGeneration.N3B3
) -> None:
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
    loihi_gen : ChipGeneration
        The generation of the Loihi board to compile. Supported
        values are N3B2, N3B3, and N3C1.
    """
    if not is_lava_loihi_installed():
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


def is_lava_loihi_installed(module_name: ty.Optional[str] = None) -> bool:
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
    lava_loihi_module = module_name or "lava.magma.compiler.subcompilers.nc"
    spec = importlib.util.find_spec(lava_loihi_module)

    return False if spec is None else True
