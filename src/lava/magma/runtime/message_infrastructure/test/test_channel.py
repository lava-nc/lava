# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from enum import Enum

from message_infrastructure import ShmemChannel
from message_infrastructure import SendPort
from message_infrastructure import RecvPort
from message_infrastructure import ChannelFactory, get_channel_factory
from message_infrastructure import ChannelTransferType
from message_infrastructure import Channel
from message_infrastructure import Selector
from message_infrastructure import SharedMemManager


def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


def main():
    channel_factory = get_channel_factory()
    data = 12
    smm = SharedMemManager()
    size = 2
    nbytes = 4
    name = 'test_shmem_channel'

    print(data)
    print(type(data))

    shmem_channel = Channel(
        ChannelTransferType.SHMEMCHANNEL,
        smm,
        size,
        nbytes,
        name)

    send_port = shmem_channel.get_send_port()
    recv_port = shmem_channel.get_recv_port()

    send_port.start()
    recv_port.start()

    # selector = Selector()
    # print(selector.select(recv_port, "cmd"))

    # send_port.send(data)
    # res = recv_port.recv()

    print("finish test function.")


main()
