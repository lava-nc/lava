# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import time
import numpy as np
from enum import Enum

from message_infrastructure import SendPort
from message_infrastructure import RecvPort
from message_infrastructure import ChannelTransferType
from message_infrastructure import Channel

def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


def main():
    data = np.array([12,24,36,48,60], dtype = np.int32)
    size = 5
    nbytes = 4 * 5
    name = 'test_shmem_channel'

    print(data)
    print(type(data))

    shmem_channel = Channel(
        ChannelTransferType.SHMEMCHANNEL,
        size,
        nbytes,
        name)

    send_port = shmem_channel.get_send_port()
    recv_port = shmem_channel.get_recv_port()

    send_port.start()
    recv_port.start()

    send_port.send(data)
    send_port.send(data)
    print("Recv data : ",recv_port.recv())
    print("Recv data : ",recv_port.recv())

    send_port.join()
    recv_port.join()

    print("finish test function.")

main()
