# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

from __future__ import annotations

import typing as ty
from dataclasses import dataclass

from lava.magma.compiler.builders.interfaces import AbstractChannelBuilder
from lava.magma.compiler.channels.watchdog import WatchdogManagerBuilder
from lava.magma.core.sync.domain import SyncDomain

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.compiler.builders.channel_builder import \
        ChannelBuilderMp
    from lava.magma.compiler.builders.runtimeservice_builder import \
        RuntimeServiceBuilder

from lava.magma.compiler.node import NodeConfig


@dataclass
class Executable:
    """Produced by compiler and contains everything the Runtime needs to run
    process.

    This includes all ProcessModels of sub processes, RuntimeService
    processes for the various nodes in the system and channel configurations.
    An Executable should be serializable so it can be saved and loaded at a
    later point.
    """
    # py_builders: ty.Dict[AbstractProcess, NcProcessBuilder]
    # c_builders: ty.Dict[AbstractProcess, CProcessBuilder]
    # nc_builders: ty.Dict[AbstractProcess, PyProcessBuilder]
    process_list: ty.List[AbstractProcess]  # All leaf processes, flat list.
    proc_builders: ty.Dict[AbstractProcess, 'AbstractProcessBuilder']
    channel_builders: ty.List[ChannelBuilderMp]
    node_configs: ty.List[NodeConfig]
    sync_domains: ty.List[SyncDomain]
    runtime_service_builders: ty.Optional[ty.Dict[SyncDomain,
                                                  RuntimeServiceBuilder]] = None
    sync_channel_builders: ty.Optional[
        ty.Iterable[AbstractChannelBuilder]] = None
    watchdog_manager_builder: WatchdogManagerBuilder = None

    def assign_runtime_to_all_processes(self, runtime):
        for p in self.process_list:
            p.runtime = runtime
