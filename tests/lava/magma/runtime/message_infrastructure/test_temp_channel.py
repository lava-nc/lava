# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from functools import partial
import time

from message_infrastructure.multiprocessing import MultiProcessing

from message_infrastructure import (
    ChannelBackend,
    Channel,
    SendPort,
    RecvPort,
    ChannelQueueSize,
    SyncChannelBytes,
    getTempRecvPort,
    getTempSendPort
)


loop_number = 1000


def prepare_data():
    return np.random.random_sample((65536, 10))


const_data = prepare_data()


def actor_stop(name):
    pass


def recv_proc(*args, **kwargs):
    actor = args[0]
    actor.set_stop_fn(partial(actor_stop, "send"))
    port = kwargs.pop("port")
    port.start()
    for i in range(loop_number):
        path, recv_port = getTempRecvPort()
        recv_port.start()
        port.send(np.array([path]))
        data = recv_port.recv()
        recv_port.join()
        if not np.array_equal(data, const_data):
            raise AssertionError()
    port.join()
    actor.status_stopped()


def send_proc(*args, **kwargs):
    actor = args[0]
    actor.set_stop_fn(partial(actor_stop, "recv"))
    port = kwargs.pop("port")
    port.start()
    for i in range(loop_number):
        path = port.recv()
        send_port = getTempSendPort(str(path[0]))
        send_port.start()
        send_port.send(const_data)
        send_port.join()
    port.join()
    actor.status_stopped()


class Builder:
    def build(self):
        pass


class TestTempChannel(unittest.TestCase):

    def test_shmemchannel(self):
        mp = MultiProcessing()
        mp.start()
        nbytes = np.prod(const_data.shape) * const_data.dtype.itemsize
        name = 'test_temp_channel'

        shmem_channel = Channel(
            ChannelBackend.SHMEMCHANNEL,
            ChannelQueueSize,
            SyncChannelBytes,
            name,
            name)

        send_port = shmem_channel.src_port
        recv_port = shmem_channel.dst_port

        recv_port_fn = partial(recv_proc, port=send_port)
        send_port_fn = partial(send_proc, port=recv_port)

        builder1 = Builder()
        builder2 = Builder()
        mp.build_actor(recv_port_fn, builder1)
        mp.build_actor(send_port_fn, builder2)

        time.sleep(0.1)
        mp.stop()
        mp.cleanup(True)


if __name__ == "__main__":
    unittest.main()
