# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np
from functools import partial

from lava.magma.runtime.message_infrastructure.MessageInfrastructurePywrapper \
    import CppMultiProcessing
from lava.magma.runtime.message_infrastructure.MessageInfrastructurePywrapper \
    import ChannelType as ChannelBackend  # noqa: E402
from lava.magma.runtime.message_infrastructure \
    import Channel, ChannelQueueSize, SyncChannelBytes
from lava.magma.runtime.message_infrastructure.interfaces import ChannelType
from lava.magma.runtime.message_infrastructure. \
    message_infrastructure_interface import MessageInfrastructureInterface

import multiprocessing as mp
import traceback

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


class SystemProcess(mp.Process):
    """Wraps a process so that the exceptions can be collected if present"""

    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None
        self._is_done = False

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    def join(self):
        if not self._is_done:
            super().join()
            super().close()
            if self._pconn.poll():
                self._exception = self._pconn.recv()
            self._cconn.close()
            self._pconn.close()
            self._is_done = True

    @property
    def exception(self):
        """Exception property."""
        if not self._is_done and self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


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
            return Channel(ChannelBackend.SHMEMCHANNEL, size,
                           channel_bytes, src_name, dst_name, shape, dtype)
        else:
            raise Exception(f"Unsupported channel type {channel_type}")
