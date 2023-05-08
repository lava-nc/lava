# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
from typing import Optional


preferred_partition: str = Optional[None]


def set_environ_settings(partition: Optional[str] = None) -> None:
    """Sets the os environment for execution on Loihi 2.
    Parameters
    ----------
    partition : str, optional
        Loihi partition name, by default None.
    """
    if 'SLURM' not in os.environ and 'NOSLURM' not in os.environ:
        os.environ['SLURM'] = '1'
    if 'LOIHI_GEN' not in os.environ:
        os.environ['LOIHI_GEN'] = 'N3B3'
    if 'PARTITION' not in os.environ and partition is not None:
        os.environ['PARTITION'] = partition


def is_available() -> bool:
    """Checks if Loihi 2 compiler is available and sets the environment
    variables.
    Returns
    -------
    bool
        Flag indicating whether Loihi 2 is available or not.
    """
    try:
        from lava.magma.compiler.subcompilers.nc.ncproc_compiler import \
            CompilerOptions
        CompilerOptions.verbose = True
    except ModuleNotFoundError:
        # Loihi 2 compiler is not available
        return False
    set_environ_settings(preferred_partition)
    return True


@property
def partition() -> str:
    """Get the partition information."""
    if 'PARTITION' in os.environ.keys():
        return os.environ['PARTITION']
    return 'Unspecified'
