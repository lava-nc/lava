# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from multiprocessing import Process as UnixProcess
from multiprocessing.managers import SharedMemoryManager

from lava.magma.compiler.channels.interfaces import ChannelType
from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.runtime.message_infrastructure.message_infrastructure_interface\
    import MessageInfrastructureInterface


class MultiProcessing(MessageInfrastructureInterface):
    """Implements message passing using shared memory and multiprocessing"""
    def __init__(self):
        self._smm: ty.Optional[SharedMemoryManager] = None
        self._actors: ty.List[UnixProcess] = []

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

    def build_actor(self, target_fn, builder) -> ty.Any:
        """Given a target_fn starts a unix process"""
        unix_process = UnixProcess(target=target_fn,
                                   args=(),
                                   kwargs={"builder": builder})
        unix_process.start()
        self._actors.append(unix_process)
        return unix_process

    def stop(self):
        """Stops the shared memory manager"""
        for actor in self._actors:
            actor.join()

        self._smm.shutdown()

    def channel_class(self, channel_type: ChannelType) -> ty.Type:
        if channel_type == ChannelType.PyPy:
            return PyPyChannel
        else:
            raise Exception(f"Unsupported channel type {channel_type}")
