# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.compiler.builders.builder import PyProcessBuilder, \
        AbstractRuntimeServiceBuilder

import multiprocessing as mp
import os
from multiprocessing.managers import SharedMemoryManager
import traceback

from lava.magma.compiler.channels.interfaces import ChannelType, Channel
from lava.magma.compiler.channels.pypychannel import PyPyChannel

from lava.magma.core.sync.domain import SyncDomain
from lava.magma.runtime.message_infrastructure.message_infrastructure_interface\
    import MessageInfrastructureInterface


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

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class MultiProcessing(MessageInfrastructureInterface):
    """Implements message passing using shared memory and multiprocessing"""

    def __init__(self):
        self._smm: ty.Optional[SharedMemoryManager] = None
        self._actors: ty.List[SystemProcess] = []

    @property
    def actors(self):
        """Returns a list of actors"""
        return self._actors

    @property
    def smm(self):
        """Returns the underlying shared memory manager"""
        return self._smm

    def start(self):
        """Starts the shared memory manager"""
        self._smm = SharedMemoryManager()
        self._smm.start()

    def build_actor(self, target_fn: ty.Callable, builder: ty.Union[
        ty.Dict['AbstractProcess', 'PyProcessBuilder'], ty.Dict[
            SyncDomain, 'AbstractRuntimeServiceBuilder']]) -> ty.Any:
        """Given a target_fn starts a system (os) process"""
        system_process = SystemProcess(target=target_fn,
                                       args=(),
                                       kwargs={"builder": builder})
        system_process.start()
        self._actors.append(system_process)
        return system_process

    def stop(self):
        """Stops the shared memory manager"""
        for actor in self._actors:
            if actor._parent_pid == os.getpid():
                actor.join()

        self._smm.shutdown()

    def channel_class(self, channel_type: ChannelType) -> ty.Type[Channel]:
        """Given a channel type, returns the shared memory based class
        implementation for the same"""
        if channel_type == ChannelType.PyPy:
            return PyPyChannel
        else:
            raise Exception(f"Unsupported channel type {channel_type}")
