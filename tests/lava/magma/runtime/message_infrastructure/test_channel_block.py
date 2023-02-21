# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from functools import partial
import time

from lava.magma.runtime.message_infrastructure import (
    PURE_PYTHON_VERSION,
    Channel,
    SendPort,
    RecvPort
)

QUEUE_SIZE = 10


def generate_data():
    return np.random.random_sample((2, 4))


def send_proc(*args, **kwargs):
    port = kwargs.pop("port")

    if not isinstance(port, SendPort):
        raise AssertionError()
    port.start()
    for i in range(QUEUE_SIZE + 1):
        data = generate_data()
        port.send(data)


def recv_proc(*args, **kwargs):
    port = kwargs.pop("port")
    port.start()
    if not isinstance(port, RecvPort):
        raise AssertionError()
    time.sleep(1)
    for i in range(QUEUE_SIZE + 1):  # pylint: disable=unused-variable
        port.recv()


class TestChannelBlock(unittest.TestCase):

    @unittest.skipIf(PURE_PYTHON_VERSION, "cpp msg lib test")
    def test_block(self):  # pylint: disable=no-self-use
        from lava.magma.runtime.message_infrastructure \
            .MessageInfrastructurePywrapper import ChannelType
        from lava.magma.runtime.message_infrastructure \
            .multiprocessing import MultiProcessing

        mp = MultiProcessing()
        mp.start()
        predata = generate_data()
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        shmem_channel = Channel(
            ChannelType.SHMEMCHANNEL,
            QUEUE_SIZE,
            nbytes,
            "test_block",
            "test_block",
            (2, 4),
            None)
        send_port = shmem_channel.src_port
        recv_port = shmem_channel.dst_port

        recv_port_fn = partial(recv_proc, port=recv_port)
        send_port_fn = partial(send_proc, port=send_port)

        mp.build_actor(recv_port_fn, None)
        mp.build_actor(send_port_fn, None)

        time.sleep(1)
        mp.stop()
        mp.cleanup(True)


if __name__ == "__main__":
    unittest.main()
