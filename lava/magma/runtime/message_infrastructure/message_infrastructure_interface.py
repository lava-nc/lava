# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC, abstractmethod

from lava.magma.compiler.channels.interfaces import ChannelType


class MessageInfrastructureInterface(ABC):
    @abstractmethod
    def start(self):
        """Starts the messaging infrastructure"""
        pass

    @abstractmethod
    def stop(self):
        """Stops the messaging infrastructure"""
        pass

    @abstractmethod
    def build_actor(self, target_fn, builder):
        """Given a target_fn starts a unix process"""
        pass

    @property
    @abstractmethod
    def actors(self) -> ty.List[ty.Any]:
        """Returns a list of actors"""
        pass

    @abstractmethod
    def channel_class(self, channel_type: ChannelType) -> ty.Type:
        pass
