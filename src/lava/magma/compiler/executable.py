# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from __future__ import annotations
import typing as ty

from lava.magma.core.sync.domain import SyncDomain

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.compiler.builders.builder import (
        PyProcessBuilder,
        CProcessBuilder,
        NcProcessBuilder,
        ChannelBuilderMp, AbstractChannelBuilder, AbstractRuntimeServiceBuilder,
    )

# ToDo: Remove when Runtime has been fixed
from lava.magma.compiler.node import NodeConfig


# ToDo: Document properly
# ToDo: Maybe it's useful to make py_builders, c_builders, nc_builders
#  dictionaries mapping from the Process to the respective builder to
#  help the Runtime, because the Builder itself (which gets shipped to a remote
#  process) does not contain a reference to the process.
class Executable:
    """Produced by compiler and contains everything the Runtime needs to run
    process.
    This includes all ProcessModels of sub processes, RuntimeService
    processes for the various nodes in the system and channel configurations.
    An Executable should be serializable so it can be saved and loaded at a
    later point.
    """

    def __init__(self):
        self.py_builders: ty.Optional[
            ty.Dict[AbstractProcess, PyProcessBuilder]] = None
        self.c_builders: ty.Optional[
            ty.Dict[AbstractProcess, CProcessBuilder]] = None
        self.nc_builders: ty.Optional[
            ty.Dict[AbstractProcess, NcProcessBuilder]] = None
        self.rs_builders: ty.Optional[
            ty.Dict[SyncDomain, AbstractRuntimeServiceBuilder]] = None
        self.sync_domains: ty.List[SyncDomain] = []
        self.node_configs: ty.List[NodeConfig] = []
        self.channel_builders: ty.List[ChannelBuilderMp] = []
        self.sync_channel_builders: ty.List[AbstractChannelBuilder] = []

    def set_py_builders(self,
                        builders: ty.Dict["AbstractProcess", PyProcessBuilder]):
        self.py_builders = builders

    def set_c_builders(self,
                       builders: ty.Dict["AbstractProcess", CProcessBuilder]):
        self.c_builders = builders

    def set_nc_builders(self,
                        builders: ty.Dict["AbstractProcess", NcProcessBuilder]):
        self.nc_builders = builders

    def set_rs_builders(self,
                        builders: ty.Dict[
                            SyncDomain, AbstractRuntimeServiceBuilder]):
        self.rs_builders = builders

    def set_sync_domains(self,
                         sync_domains: ty.List[SyncDomain]):
        self.sync_domains = sync_domains

    def set_node_cfgs(self,
                      node_cfgs: ty.List[NodeConfig]):
        self.node_configs = node_cfgs

    def set_channel_builders(self,
                             channel_builders: ty.List[ChannelBuilderMp]):
        self.channel_builders = channel_builders

    def set_sync_channel_builders(self,
                                  sync_channel_builders: ty.List[
                                      AbstractChannelBuilder]):
        self.sync_channel_builders = sync_channel_builders
