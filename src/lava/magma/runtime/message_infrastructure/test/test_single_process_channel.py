# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import time
import numpy as np
from enum import Enum

from message_infrastructure.multiprocessing import MultiProcessing

from message_infrastructure import (
    ChannelBackend,
    Channel,
    SendPort,
    RecvPort
)


def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


def main():
    data = np.array([12, 24, 36, 48, 60], dtype=np.int32)
    print("Send data: ", data)

    size = 5
    nbytes = nbytes_cal(data.shape, data.dtype)
    name = 'test_shmem_channel'

    shmem_channel = Channel(
        ChannelBackend.SHMEMCHANNEL,
        size,
        nbytes,
        name)

    send_port = shmem_channel.get_send_port()
    recv_port = shmem_channel.get_recv_port()

    send_port.start()
    recv_port.start()

    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    res = recv_port.recv()
    assert np.array_equal(res, data)
    res = recv_port.recv()
    assert np.array_equal(res, data)
    res = recv_port.recv()
    assert np.array_equal(res, data)
    res = recv_port.recv()
    assert np.array_equal(res, data)

    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)

    res = recv_port.recv()
    assert np.array_equal(res, data)
    res = recv_port.recv()
    assert np.array_equal(res, data)

    print("Recv data: ", res)

    print("finish test function.")

    send_port.join()
    recv_port.join()


main()
