# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod

from lava.magma.compiler.builders.interfaces import AbstractProcessBuilder
from lava.magma.compiler.channel_map import ChannelMap
from lava.magma.compiler.compiler_graphs import ProcGroup
from lava.magma.core.process.process import AbstractProcess


class AbstractSubCompiler(ABC):
    """Interface for SubCompilers. Their job is to compile connected groups of
    Processes, whose ProcessModels can be executed on the same type of
    backend."""

    @abstractmethod
    def compile(self, channel_map: ChannelMap) -> ChannelMap:
        """Partitions all Processes in the SubCompiler's ProcGroup onto the
        available resources."""

    @abstractmethod
    def get_builders(
        self, channel_map: ChannelMap
    ) -> ty.Tuple[ty.Dict[AbstractProcess, AbstractProcessBuilder], ChannelMap]:
        """After compilation, creates and returns builders for all Processes."""


class SubCompiler(AbstractSubCompiler):
    def __init__(
        self,
        proc_group: ty.Optional[ProcGroup],
        compile_config: ty.Optional[ty.Dict[str, ty.Any]] = None,
    ):
        """SubCompiler that already has some implementation details but is
        otherwise abstract.

        Parameters
        ----------
        proc_group : ProcGroup
            Group of Processes that will be compiled by this SubCompiler.
        compile_config : dict(str, Any), optional
            Dictionary containing configuration options for the CProcCompiler.
        """
        self._compile_config = compile_config or {}
        self._proc_group = proc_group
        self._tmp_channel_map: ty.Optional[ChannelMap] = None

    @abstractmethod
    def compile(self, channel_map: ChannelMap) -> ChannelMap:
        pass

    @abstractmethod
    def get_builders(
        self, channel_map: ChannelMap
    ) -> ty.Tuple[ty.Dict[AbstractProcess, AbstractProcessBuilder], ChannelMap]:
        pass
