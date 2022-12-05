# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC, abstractmethod
from message_infrastructure import ChannelBackend
from message_infrastructure import Channel

"""A Message Infrastructure Interface which can create actors which would
participate in message passing/exchange, start and stop them as well as
declare the underlying Channel Infrastructure Class to be used for message
passing implementation."""


class MessageInfrastructureInterface(ABC):
    """Interface to provide the ability to create actors which can
    communicate via message passing"""
    @abstractmethod
    def start(self):
        """Starts the messaging infrastructure"""
        pass

    @abstractmethod
    def stop(self):
        """Stops the messaging infrastructure"""
        pass

    @abstractmethod
    def build_actor(self, target_fn: ty.Callable, builder):
        """Given a target_fn starts a system process"""
        pass

    @property
    @abstractmethod
    def actors(self) -> ty.List[ty.Any]:
        """Returns a list of actors"""
        pass

    @abstractmethod
    def channel_class(self,
                      channel_type: ChannelBackend) -> ty.Type[Channel]:
        """Given the Channel Type, Return the Channel Implementation to
        be used during execution"""
        pass
