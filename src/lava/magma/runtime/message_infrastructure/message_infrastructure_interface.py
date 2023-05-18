# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.compiler.builders.py_builder import PyProcessBuilder
    from lava.magma.compiler.builders.runtimeservice_builder import \
        RuntimeServiceBuilder
from abc import ABC, abstractmethod

from lava.magma.compiler.channels.interfaces import ChannelType, Channel
from lava.magma.core.sync.domain import SyncDomain


class MessageInfrastructureInterface(ABC):
    """A Message Infrastructure Interface which can create actors which would
    participate in message passing/exchange, start and stop them as well as
    declare the underlying Channel Infrastructure Class to be used for message
    passing implementation."""

    @abstractmethod
    def start(self):
        """Starts the messaging infrastructure"""

    @abstractmethod
    def stop(self):
        """Stops the messaging infrastructure"""

    @abstractmethod
    def build_actor(self, target_fn: ty.Callable, builder: ty.Union[
        ty.Dict['AbstractProcess', 'PyProcessBuilder'], ty.Dict[
            SyncDomain, 'RuntimeServiceBuilder']]):
        """Given a target_fn starts a system process"""

    @property
    @abstractmethod
    def actors(self) -> ty.List[ty.Any]:
        """Returns a list of actors"""

    @abstractmethod
    def channel_class(self, channel_type: ChannelType) -> ty.Type[Channel]:
        """Given the Channel Type, Return the Channel Implementation to
        be used during execution"""
