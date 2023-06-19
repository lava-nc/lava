# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod
from lava.magma.runtime.message_infrastructure import Channel
from lava.magma.runtime.message_infrastructure.interfaces import ChannelType


class MessageInfrastructureInterface(ABC):
    """Interface to provide the ability to create actors which can
    communicate via message passing"""

    @abstractmethod
    def init(self):
        """Init the messaging infrastructure"""
        pass

    @abstractmethod
    def start(self):
        """Starts the messaging infrastructure"""

    def pre_stop(self):
        """Stop MessageInfrastructure before join ports"""
        pass

    def stop(self):
        """Stops the messaging infrastructure after join ports"""
        pass

    @abstractmethod
    def build_actor(self, target_fn: ty.Callable, builder):
        """Given a target_fn starts a system process"""

    def cleanup(self, block=False):
        """Close all resources"""
        pass

    def trace(self, logger) -> int:
        """Trace actors' exceptions"""
        return 0

    @property
    @abstractmethod
    def actors(self) -> ty.List[ty.Any]:
        """Returns a list of actors"""

    @abstractmethod
    def channel(self, channel_type: ChannelType, src_name, dst_name,
                shape, dtype, size, sync=False) -> Channel:
        """Given the Channel Type, Return the Channel Implementation to
        be used during execution"""
