# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.compiler.builders.interfaces import AbstractProcessBuilder
from lava.magma.compiler.builders.py_builder import PyProcessBuilder

try:
    from lava.magma.compiler.builders.c_builder import CProcessBuilder
    from lava.magma.compiler.builders.nc_builder import NcProcessBuilder
except ImportError:
    class CProcessBuilder(AbstractProcessBuilder):
        pass

    class NcProcessBuilder(AbstractProcessBuilder):
        pass

from lava.magma.core.process.process import AbstractProcess


def split_proc_builders_by_type(proc_builders: ty.Dict[AbstractProcess,
                                                       AbstractProcessBuilder]):
    """Given a dictionary of process to builders, returns a tuple of
    process to builder dictionaries for Py, C and Nc processes."""
    py_builders = {}
    c_builders = {}
    nc_builders = {}
    for proc, builder in proc_builders.items():
        entry = {proc: builder}
        if isinstance(builder, PyProcessBuilder):
            py_builders.update(entry)
        elif isinstance(builder, CProcessBuilder):
            c_builders.update(entry)
        elif isinstance(builder, NcProcessBuilder):
            nc_builders.update(entry)
        else:
            raise TypeError(
                f"The builder of type {type(builder)} is not "
                f"supported by the Executable.")
    return py_builders, c_builders, nc_builders
