# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
import typing as ty
from functools import partial

from message_infrastructure import CppMultiProcessing
from message_infrastructure import SharedMemManager
from message_infrastructure import Actor
from message_infrastructure import ChannelBackend
from message_infrastructure import Channel

from message_infrastructure.message_infrastructure_interface \
    import MessageInfrastructureInterface


"""Implements the Message Infrastructure Interface using Python
MultiProcessing Library. The MultiProcessing API is used to create actors
which will participate in exchanging messages. The Channel Infrastructure
further uses the SharedMemoryManager from MultiProcessing Library to
implement the communication backend in this implementation."""


class MultiProcessing(MessageInfrastructureInterface):
    """Implements message passing using shared memory and multiprocessing"""

    def __init__(self):
        self._mp: ty.Optional[CppMultiProcessing] = CppMultiProcessing()

    @property
    def actors(self):
        """Returns a list of actors"""
        return self._mp.get_actors()

    @property
    def smm(self):
        """Returns the underlying shared memory manager"""
        return self._mp.get_shmm()

    def start(self):
        """Init the MultiProcessing"""
        pass

    def build_actor(self, target_fn: ty.Callable, builder) -> ty.Any:
        """Given a target_fn starts a system (os) process"""
        bound_target_fn = partial(target_fn, builder=builder)
        ret = self._mp.build_actor(bound_target_fn)

    def stop(self):
        """Stops the shared memory manager"""
        self._mp.stop()

    def channel_class(self,
                      channel_type: ChannelBackend) -> ty.Type[Channel]:
        """TODO: depricated. Return None"""
        return None
