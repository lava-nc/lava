# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import importlib.util


def is_installed() -> bool:
    """Returns whether the Lava extension for Loihi is installed.

    Returns
    -------
    bool
        True iff lava-loihi can be imported in this Python environment.
    """
    lava_loihi_module = "lava.magma.compiler.subcompilers.nc"
    spec = importlib.util.find_spec(lava_loihi_module)

    return True if spec is not None else False

