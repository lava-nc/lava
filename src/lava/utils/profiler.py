# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.run_configs import RunConfig, Loihi2HwCfg


class Profiler:
    """Base class for profiling execution time, energy and other
    metrics on different Ressources. Depending on the computing
    ressource an appropriate profiler needs to be chosen. The run
    configuration is used to choose the related profiler, if there
    is one."""

    run_cfg_dict = {Loihi2HwCfg: Loihi2HWProfiler}

    @staticmethod
    def init(run_cfg: RunConfig):
        """Decide which profiler is needed based on the run
        configuration."""

        if not isinstance(run_cfg, RunConfig):
            raise AssertionError("<run_cfg> must be an "
                                 "instance of {}".format(RunConfig))

        if not type(run_cfg) in Profiler.run_cfg_dict:
            raise NotImplementedError(
                f"There is currently no implementation of the "
                f"profiler for {type(run_cfg).__name__}.")

        return Profiler.run_cfg_dict[type(run_cfg)](run_cfg)
