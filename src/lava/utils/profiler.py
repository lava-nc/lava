# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import types
import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.run_conditions import AbstractRunCondition
from lava.magma.core.run_configs import RunConfig
from lava.magma.runtime.runtime import Runtime
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.resources import (
    AbstractComputeResource, Loihi1NeuroCore, Loihi2NeuroCore)


class Profiler:
    """The Profiler is a tool to determine power and performance of workloads
    for Loihi 1 or Loihi 2 based on software simulations or hardware
    measurements.

    The execution time and energy of a workload is either measured on hardware
    during execution or estimated in simulation. The estimation is based on
    elementary hardware operations which are counted during the simulation.
    Each elementary operation has a defined execution time and energy cost,
    which is used in a performance model to calculate execution time and energy.
    """

    def __init__(self, start: int = 0, end: int = 0,
                 bin_size: int = 1, buffer_size: int = 1000):
        self.start = start
        self.end = end
        self.bin_size = bin_size
        self.buffer_size = buffer_size
        self.used_resources: ty.List[AbstractComputeResource] = []

    def profile(self, proc: AbstractProcess):
        proc.run = types.MethodType(self.run, proc)

    def get_energy(self) -> np.array:
        """Returns the energy estimate per time step in µJ."""
        ...
        return 0

    def get_power(self) -> np.array:
        """Returns the power estimate per time step in µW."""
        ...
        return 0

    def get_execution_time(self) -> np.array:
        """Returns the execution time estimate per time step in µs."""
        ...
        return 0

    def run(self, proc: AbstractProcess, condition: AbstractRunCondition = None,
            run_cfg: RunConfig = None):
        """Runs process given RunConfig and RunCondition.

        Functionally, this method does the same as run(..) of AbstractProcess,
        but modifies the chosen ProcModels and executables to be able to use the
        Profiler. From the user perspective, it should not be noticeable as
        the API does not change. This method will be used to override the method
        run(..) of an instance of AbstractProcess, when the Profiler is
        created.

        Parameters
        ----------
        proc : AbstractProcess
            Process instance which run(..) was initially called on.
        condition : AbstractRunCondition
            RunCondition instance specifies for how long to run the process.
        run_cfg : RunConfig
            RunConfig is used by compiler to select a ProcessModel for each
            compiled process.
        """

        if not proc._runtime:

            compiler = Compiler(loglevel=proc.loglevel)
            # initializer = Initializer()

            # 1. get proc_map
            # proc_map = initializer._map_proc_to_model(proc, run_cfg)
            proc_map = compiler._map_proc_to_model(
                compiler._find_processes(proc), run_cfg)

            # 2. modify proc_map
            proc_map = self._modify_proc_map(proc_map)

            # 3.  prepare ProcModels for profiling
            self._prepare_proc_models(proc_map)

            # 4. create executable
            executable = compiler.compile(proc, run_cfg)

            # 5. append profiler sync channels
            self._set_profiler_sync_channel_builders(executable)

            # 6. create Runtime
            proc._runtime = Runtime(executable,
                                    ActorType.MultiProcessing,
                                    loglevel=proc.loglevel)
            proc._runtime.initialize()

        proc._runtime.start(condition)

    def _modify_proc_map(self, proc_map):
        """Check if chosen process models have a profileable version and
        exchange the process models accordingly.
        Tell the user which Processes will not be profiled, as they lack a
        profileable ProcModel."""
        ...
        return proc_map

    def _prepare_proc_models(self, proc_map):
        """Prepare each ProcModel for profiling.
        Configure Monitors for ProcModels executing in simulation.
        Recognize if ProcModels execute on Hardware."""
        for proc_model, proc in proc_map.items():
            if Loihi1NeuroCore in proc.required_resources:
                self.used_resources.append(Loihi1NeuroCore)
            else:
                # 1. add operation counter Vars to the Process
                # 2. set up Monitors to operation counter Vars
                ...

    def _set_profiler_sync_channel_builders(self, executable):
        """Create and append sync_channel builders if Loihi compute node is
        going to execute a profileable ProcModel."""
        if Loihi1NeuroCore in self.used_resources or \
                Loihi2NeuroCore in self.used_resources:
            executable.sync_channel_builders.append(...)
        ...
