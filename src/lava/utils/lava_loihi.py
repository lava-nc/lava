# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import importlib.util


def is_installed(module_name: ty.Optional[str] = None) -> bool:
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
