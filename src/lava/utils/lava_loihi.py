# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


def is_installed() -> bool:
    """Returns whether the Lava extension for Loihi is installed.

    Returns
    -------
    bool
        True iff lava-loihi can be imported in this python environment.
    """
    try:
        from lava.magma.compiler.subcompilers.nc.ncproc_compiler import \
            NcProcCompiler
    except ModuleNotFoundError:
        return False

    return True
