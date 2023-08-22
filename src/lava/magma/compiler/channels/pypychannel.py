# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
from dataclasses import dataclass
from multiprocessing import Semaphore
from queue import Queue, Empty
from threading import BoundedSemaphore, Condition, Thread
from time import time
from scipy.sparse import csr_matrix
from lava.utils.sparse import find
from lava.magma.compiler.channels.watchdog import Watchdog, NoOPWatchdog


import numpy as np
from lava.magma.compiler.channels.interfaces import (
    Channel,
    AbstractCspSendPort,
    AbstractCspRecvPort,
)

if ty.TYPE_CHECKING:
    from lava.magma.runtime.message_infrastructure \
        .message_infrastructure_interface import (
            MessageInfrastructureInterface)


@dataclass
class Proto:
    shape: np.ndarray
    dtype: np.dtype
    nbytes: int


class CspSendPort(AbstractCspSendPort):
    """
    CspSendPort is a low level send port implementation based on CSP
    semantics. It can be understood as the input port of a CSP channel.
    """

    def __init__(self, name, shm, proto, size, req, ack,
                 io_watchdog: Watchdog, join_watchdog: Watchdog):
        """Instantiates CspSendPort object and class attributes

        Parameters
        ----------
        name : str
        shm : SharedMemory
        proto : Proto
        size : int
        req : Semaphore
        ack : Semaphore
        """
        self._name = name
        self._shm = shm
        self._shape = proto.shape
        self._dtype = proto.dtype
        self._nbytes = proto.nbytes
        self._req = req
        self._ack = ack
        self._size = size
        self._idx = 0
        self._done = False
        self._array = []
        self._semaphore = None
        self.observer: ty.Optional[ty.Callable[[], ty.Any]] = None
        self.thread = None

        self._io_watchdog: Watchdog = io_watchdog
        self._join_watchdog: Watchdog = join_watchdog

    @property
    def name(self) -> str:
        return self._name

    @property
    def d_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return self._size

    def start(self):
        """Starts the port to listen on a thread"""
        self._array = [
            np.ndarray(
                shape=self._shape,
                dtype=self._dtype,
                buffer=self._shm.buf[
                    self._nbytes * i: self._nbytes * (i + 1)
                ],
            )
            for i in range(self._size)
        ]
        self._semaphore = BoundedSemaphore(self._size)
        self.thread = Thread(
            target=self._ack_callback,
            name="{}.send".format(self._name),
            args=(self._ack,),
            daemon=True,
        )
        self.thread.start()

    def _ack_callback(self, ack):
        try:
            while ack.acquire() and not self._done:
                not_full = self.probe()
                self._semaphore.release()
                if self.observer and not not_full:
                    self.observer()
        except EOFError:
            pass

    def probe(self):
        """
        Returns True if a 'send' call will not block, and False otherwise.
        Does not block.
        """
        result = self._semaphore.acquire(blocking=False)
        if result:
            self._semaphore.release()
        return result

    def send(self, data):
        """
        Send data on the channel. May block if the channel is already full.
        """
        with self._io_watchdog:
            if data.shape != self._shape:
                raise AssertionError(f"{data.shape=} {self._shape=} Mismatch")

            if isinstance(data, csr_matrix):
                data = find(data, explicit_zeros=True)[2]

            self._semaphore.acquire()
            self._array[self._idx][:] = data[:]
            self._idx = (self._idx + 1) % self._size
            self._req.release()

    def join(self):
        with self._join_watchdog:
            if not self._done:
                self._done = True
                if self.thread is not None:
                    self._ack.release()
                self._ack = None
                self._req = None


class CspRecvQueue(Queue):
    """
    Underlying queue which backs the CspRecvPort
    """

    def get(self, block=True, timeout=None, peek=False):
        """
        Implementation from the standard library augmented with 'peek' to
        optionally return the head element without removing it.
        """
        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self._qsize():
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            if peek:
                item = self.queue[0]
            else:
                item = self._get()
                self.not_full.notify()
            return item


class CspRecvPort(AbstractCspRecvPort):
    """
    CspRecvPort is a low level recv port implementation based on CSP
    semantics. It can be understood as the output port of a CSP channel.
    """

    def __init__(self, name, shm, proto, size, req, ack,
                 io_watchdog, join_watchdog):
        """Instantiates CspRecvPort object and class attributes

        Parameters
        ----------
        name : str
        shm : SharedMemory
        proto : Proto
        size : int
        req : Semaphore
        ack : Semaphore
        """
        self._name = name
        self._shm = shm
        self._shape = proto.shape
        self._dtype = proto.dtype
        self._nbytes = proto.nbytes
        self._size = size
        self._req = req
        self._ack = ack
        self._idx = 0
        self._done = False
        self._array = []
        self._queue = None
        self.observer: ty.Optional[ty.Callable[[], ty.Any]] = None
        self.thread = None

        self._io_watchdog = io_watchdog
        self._join_watchdog = join_watchdog

    @property
    def name(self) -> str:
        return self._name

    @property
    def d_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return self._size

    def start(self):
        """Starts the port to listen on a thread"""
        # LavaCDataType.INT32 is equal to 4
        if self._dtype == 4:
            self._dtype = np.int32
        self._array = [
            np.ndarray(
                shape=self._shape,
                dtype=self._dtype,
                buffer=self._shm.buf[
                    self._nbytes * i: self._nbytes * (i + 1)
                ],
            )
            for i in range(self._size)
        ]
        self._queue = CspRecvQueue(self._size)
        self.thread = Thread(
            target=self._req_callback,
            name="{}.send".format(self._name),
            args=(self._req,),
            daemon=True,
        )
        self.thread.start()

    def _req_callback(self, req):
        try:
            while req.acquire() and not self._done:
                not_empty = self.probe()
                self._queue.put_nowait(0)
                if self.observer and not not_empty:
                    self.observer()
        except EOFError:
            pass

    def probe(self):
        """
        Returns True if a 'recv' call will not block, and False otherwise.
        Does not block.
        """
        return self._queue.qsize() > 0

    def peek(self):
        """
        Return the next token on the channel without acknowledging it. Blocks
        if there is no data on the channel.
        """
        self._queue.get(peek=True)
        result = self._array[self._idx].copy()
        return result

    def recv(self):
        """
        Receive from the channel. Blocks if there is no data on the channel.
        """
        with self._io_watchdog:
            self._queue.get()
            result = self._array[self._idx].copy()
            self._idx = (self._idx + 1) % self._size
            self._ack.release()
            return result

    def join(self):
        with self._join_watchdog:
            if not self._done:
                self._done = True
                if self.thread is not None:
                    self._req.release()
                self._ack = None
                self._req = None


class CspSelector:
    """
    Utility class to allow waiting for multiple channels to become ready
    """

    def __init__(self):
        """Instantiates CspSelector object and class attributes"""
        self._cv = Condition()

    def _changed(self):
        with self._cv:
            self._cv.notify_all()

    @staticmethod
    def _set_observer(
            channel_actions: ty.Tuple,
            observer: ty.Union[ty.Callable[[], ty.Any], None]) -> None:
        for channel, _ in channel_actions:
            channel.observer = observer

    def select(
            self,
            *channel_actions: ty.Tuple[
                ty.Union[CspSendPort, CspRecvPort],
                ty.Callable[[], ty.Any]
            ],
    ) -> None:
        """
        Wait for any channel to become ready, then execute the corresponding
        callable and return the result.
        """
        with self._cv:
            self._set_observer(channel_actions, self._changed)
            while True:
                for channel, action in channel_actions:
                    if channel.probe():
                        self._set_observer(channel_actions, None)
                        return action()
                self._cv.wait()


class PyPyChannel(Channel):
    """Helper class to create the set of send and recv port and encapsulate
    them inside a common structure. We call this a PyPyChannel"""

    def __init__(
            self,
            message_infrastructure: "MessageInfrastructureInterface",
            src_name,
            dst_name,
            shape,
            dtype,
            size,
            src_send_watchdog=NoOPWatchdog(None),
            src_join_watchdog=NoOPWatchdog(None),
            dst_recv_watchdog=NoOPWatchdog(None),
            dst_join_watchdog=NoOPWatchdog(None)
    ):
        """Instantiates PyPyChannel object and class attributes

        Parameters
        ----------
        message_infrastructure: MessageInfrastructureInterface
        src_name : str
        dst_name : str
        shape : ty.Tuple[int, ...]
        dtype : ty.Type[np.intc]
        size : int
        """
        nbytes = self.nbytes(shape, dtype)
        smm = message_infrastructure.smm
        shm = smm.create_shared_memory(int(nbytes * size))
        req = Semaphore(0)
        ack = Semaphore(0)
        proto = Proto(shape=shape, dtype=dtype, nbytes=nbytes)
        self._src_port = CspSendPort(src_name, shm, proto, size, req, ack,
                                     src_send_watchdog,
                                     src_join_watchdog)
        self._dst_port = CspRecvPort(dst_name, shm, proto, size, req, ack,
                                     dst_recv_watchdog,
                                     dst_join_watchdog)

    def nbytes(self, shape, dtype):
        return np.prod(shape) * np.dtype(dtype).itemsize

    @property
    def src_port(self) -> AbstractCspSendPort:
        return self._src_port

    @property
    def dst_port(self) -> AbstractCspRecvPort:
        return self._dst_port
