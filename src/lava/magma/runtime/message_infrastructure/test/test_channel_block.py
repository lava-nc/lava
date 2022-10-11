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
    RecvPort
)

QUEUE_SIZE = 10


def generate_data():
    return np.random.random_sample((2, 4))


def actor_stop(port, name):
    port.join()


def send_proc(actor, **kwargs):
    port = kwargs.pop("port")
    actor.set_stop_fn(partial(actor_stop, port, "send"))
    if not isinstance(port, SendPort):
        raise AssertionError()
    port.start()
    for i in range(QUEUE_SIZE + 1):
        start_ts = time.time()
        data = generate_data()
        port.send(data)
        end_ts = time.time()
    actor.status_paused()


def recv_proc(actor, **kwargs):
    port = kwargs.pop("port")
    actor.set_stop_fn(partial(actor_stop, port, "recv"))
    port.start()
    if not isinstance(port, RecvPort):
        raise AssertionError()
    time.sleep(1)
    for i in range(QUEUE_SIZE + 1):
        data = port.recv()
    actor.status_paused()


class TestChannelBlock(unittest.TestCase):

    def test_block(self):
        mp = MultiProcessing()
        mp.start()
        predata = generate_data()
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        shmem_channel = Channel(
            ChannelBackend.SHMEMCHANNEL,
            QUEUE_SIZE,
            nbytes,
            "test_block",
            "test_block")

        send_port = shmem_channel.src_port
        recv_port = shmem_channel.dst_port

        recv_port_fn = partial(recv_proc, port=recv_port)
        send_port_fn = partial(send_proc, port=send_port)

        mp.build_actor(recv_port_fn, None)
        mp.build_actor(send_port_fn, None)

        time.sleep(1)
        mp.stop(True)


if __name__ == "__main__":
    unittest.main()
