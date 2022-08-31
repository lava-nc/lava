# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.compiler.channels.interfaces import ChannelType
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.runtime.message_infrastructure \
    .message_infrastructure_interface import \
    MessageInfrastructureInterface


class NxBoardMsgInterface(MessageInfrastructureInterface):
    """Implements message passing using nx board"""

    @property
    def actors(self):
        """Returns a list of actors"""
        pass

    def start(self):
        """Starts the shared memory manager"""
        pass

    def build_actor(self, target_fn: ty.Callable, builder: ty.Union[
        ty.Dict['AbstractProcess', 'PyProcessBuilder'], ty.Dict[
            SyncDomain, 'RuntimeServiceBuilder']]) -> ty.Any:
        """Given a target_fn starts a system (os) process"""
        pass

    def stop(self):
        """Stops the shared memory manager"""
        pass

    def channel_class(self, channel_type: ChannelType) -> ty.Type[ChannelType]:
        """Given a channel type, returns the shared memory based class
        implementation for the same."""
        if channel_type == ChannelType.CNc:
            return CNcChannel
        elif channel_type == ChannelType.NcC:
            return NcCChannel
        else:
            raise Exception(f"Unsupported channel type {channel_type}")
