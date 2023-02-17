# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
import typing as ty
import numpy as np
from functools import partial

from lava.magma.runtime.message_infrastructure.MessageInfrastructurePywrapper \
    import (CppMultiProcessing,
            Actor)
from lava.magma.runtime.message_infrastructure.MessageInfrastructurePywrapper \
    import ChannelType as ChannelBackend  # noqa: E402
from lava.magma.runtime.message_infrastructure \
    import Channel, ChannelQueueSize, SyncChannelBytes
from lava.magma.runtime.message_infrastructure.interfaces import ChannelType
from lava.magma.runtime.message_infrastructure. \
    message_infrastructure_interface import MessageInfrastructureInterface

try:
    from lava.magma.core.model.c.type import LavaTypeTransfer
except ImportError:
    class LavaTypeTransfer:
        pass


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

    def init(self):
        pass

    def start(self):
        """Init the MultiProcessing"""
        for actor in self._mp.get_actors():
            actor.start()

    def build_actor(self, target_fn: ty.Callable, builder) -> ty.Any:
        """Given a target_fn starts a system (os) process"""
        bound_target_fn = partial(target_fn, builder=builder)
        self._mp.build_actor(bound_target_fn)

    def pre_stop(self):
        """Stops the shared memory manager"""
        self._mp.stop()

    def pause(self):
        for actor in self._mp.get_actors():
            actor.pause()

    def cleanup(self, block=False):
        """Close all resources"""
        self._mp.cleanup(block)

    def trace(self, logger) -> int:
        """Trace actors' exceptions"""
        # CppMessageInfrastructure cannot trace exceptions.
        # It needs to stop all actors.
        self.stop()
        return 0

    def channel(self, channel_type: ChannelType, src_name, dst_name,
                shape, dtype, size, sync=False) -> Channel:
        if channel_type == ChannelType.PyPy:
            channel_bytes = np.prod(shape) * np.dtype(dtype).itemsize \
                if not sync else SyncChannelBytes
            return Channel(ChannelBackend.SHMEMCHANNEL, ChannelQueueSize,
                           channel_bytes, src_name, dst_name, shape, dtype)
        elif channel_type == ChannelType.PyC or channel_type == ChannelType.CPy:
            temp_dtype = LavaTypeTransfer.cdtype2numpy(dtype)
            channel_bytes = np.prod(shape) * np.dtype(temp_dtype).itemsize \
                if not sync else SyncChannelBytes
            return Channel(ChannelBackend.SHMEMCHANNEL, ChannelQueueSize,
                           channel_bytes, src_name, dst_name, shape, dtype)
        else:
            raise Exception(f"Unsupported channel type {channel_type}")
