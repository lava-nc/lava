# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from enum import Enum

from message_infrastructure import ShmemChannel
from message_infrastructure import SendPort
from message_infrastructure import RecvPort
from message_infrastructure import ChannelFactory, get_channel_factory
from message_infrastructure import SharedMemory
from message_infrastructure import ChannelBackend
from message_infrastructure import Channel
from message_infrastructure import Selector


def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


def main():
    channel_factory = get_channel_factory()
    data = np.array([1, 2, 3], np.int32)
    shm = SharedMemory()
    size = 2
    nbytes = nbytes_cal(data.shape, data.dtype)
    name = 'test_channel'

    print(data)
    print(type(data))
    print(data.dtype)
    print(ChannelBackend.SHMEMCHANNEL)
    print(type(ChannelBackend.SHMEMCHANNEL))

    shmem_channel = channel_factory.get_channel(
        ChannelBackend.SHMEMCHANNEL,
        shm,
        data,
        size,
        nbytes,
        name)

    send_port = shmem_channel.get_send_port()
    recv_port = shmem_channel.get_recv_port()

    send_port.start()
    recv_port.start()

    selector = Selector()
    print(selector.select(recv_port, "cmd"))

    print("finish test function.")


main()
