# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import sys
import warnings
import typing as ty
from lava.magma.core.run_configs import RunConfig, Loihi2HwCfg

# Dictionary to relate a certain RunCfg to specific Profiler
run_cfg_dict: ty.Dict[RunConfig, 'Profiler'] = {}

try:
    from lava.utils.loihi2_profiler_api import Loihi2HWProfiler
    run_cfg_dict[Loihi2HwCfg] = Loihi2HWProfiler
except ModuleNotFoundError:
    warnings.warn("Loihi2HWProfiler could not be imported. "
                  "Currently no profiler is available.")


def get_pyobj_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_pyobj_size(v, seen) for v in obj.values()])
        size += sum([get_pyobj_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_pyobj_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_pyobj_size(i, seen) for i in obj])
    return size


class Profiler:
    """Base class for profiling execution time, energy and other
    metrics on different resources. Depending on the computing
    ressource an appropriate profiler needs to be chosen. The run
    configuration is used to choose the related profiler, if there
    is one."""

    @staticmethod
    def init(run_cfg: RunConfig) -> 'Profiler':
        """Decide which profiler is needed based on the run
        configuration."""

        if not isinstance(run_cfg, RunConfig):
            raise TypeError("<run_cfg> must be an "
                            "instance of {}".format(RunConfig))

        if not type(run_cfg) in run_cfg_dict:
            raise NotImplementedError(
                f"There is currently no implementation of the "
                f"profiler for {type(run_cfg).__name__}.")

        return run_cfg_dict[type(run_cfg)](run_cfg)
