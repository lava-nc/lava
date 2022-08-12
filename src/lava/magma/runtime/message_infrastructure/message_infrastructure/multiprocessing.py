# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
from pty import CHILD
import typing as ty
if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.compiler.builders.py_builder import PyProcessBuilder
    from lava.magma.compiler.builders.runtimeservice_builder import \
        RuntimeServiceBuilder

from MessageInfrastructurePywrapper import CppMultiProcessing
from MessageInfrastructurePywrapper import SharedMemoryManager
from MessageInfrastructurePywrapper import Actor

from enum import Enum 

from lava.magma.compiler.channels.interfaces import ChannelType, Channel
from lava.magma.compiler.channels.pypychannel import PyPyChannel

try:
    from lava.magma.compiler.channels.cpychannel import \
        CPyChannel, PyCChannel
except ImportError:
    class CPyChannel:
        pass

    class PyCChannel:
        pass

from lava.magma.core.sync.domain import SyncDomain
from lava.magma.runtime.message_infrastructure.message_infrastructure\
    .message_infrastructure_interface import MessageInfrastructureInterface


"""Implements the Message Infrastructure Interface using Python
MultiProcessing Library. The MultiProcessing API is used to create actors
which will participate in exchanging messages. The Channel Infrastructure
further uses the SharedMemoryManager from MultiProcessing Library to
implement the communication backend in this implementation."""


class ProcessType(Enum):
    ERR_PROC = -1
    PARENT_PROC = 0
    CHILD_PROC = 1


class MultiProcessing(MessageInfrastructureInterface):
    """Implements message passing using shared memory and multiprocessing"""

    def __init__(self):
        self._mp: ty.Optional[CppMultiProcessing] = None
        self._smm: ty.Optional[SharedMemoryManager] = None
        self._actors: ty.List[Actor] = []

    @property
    def actors(self):
        """Returns a list of actors"""
        return self._mp.get_actors()

    @property
    def smm(self):
        """Returns the underlying shared memory manager"""
        return self._smm

    def start(self):
        """Starts the shared memory manager"""
        self._mp = CppMultiProcessing()
        self._smm = SharedMemoryManager()

    def build_actor(self, target_fn: ty.Callable, builder: ty.Union[
        ty.Dict['AbstractProcess', 'PyProcessBuilder'], ty.Dict[
            SyncDomain, 'RuntimeServiceBuilder']]) -> ty.Any:
        """Given a target_fn starts a system (os) process"""

        system_process = SystemProcess(target=target_fn,
                                       args=(),
                                       kwargs={"builder": builder})
        ret = self._mp.build_actor()
        if ret == ProcessType.ERR_PROC:
            exit(-1)
        if ret == ProcessType.CHILD_PROC:
            target_fn(args=(), kwargs={"builder": builder})
            exit(0)

    def stop(self):
        """Stops the shared memory manager"""
        self._mp.stop()
        self._smm.stop()

    def channel_class(self, channel_type: ChannelType) -> ty.Type[Channel]:
        """Given a channel type, returns the shared memory based class
        implementation for the same"""
        if channel_type == ChannelType.PyPy:
            return PyPyChannel
        elif channel_type == ChannelType.PyC:
            return PyCChannel
        elif channel_type == ChannelType.CPy:
            return CPyChannel
        else:
            raise Exception(f"Unsupported channel type {channel_type}")
