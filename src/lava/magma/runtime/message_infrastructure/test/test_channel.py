# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import re
import numpy as np
import unittest
import traceback
from functools import partial
import time

<<<<<<< HEAD
from message_infrastructure.multiprocessing import MultiProcessing
=======

def nbytes_cal(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize
>>>>>>> 8ffebb14fd29b0af6792065cd24354c7b7132ac1

from message_infrastructure import (
    ChannelTransferType,
    Channel,
    SendPort,
    RecvPort
)

<<<<<<< HEAD
def prepare_data():
    data = np.array([12,24,36,48,60], dtype = np.int32)
    return data

def send_proc(*args, **kwargs):
    try:
        actor = args[0]
        port = kwargs.pop("port")
        assert isinstance(port, SendPort)
        port.start()
        port.send(prepare_data())
    except Exception as e:
        print("send error")
        raise e

def recv_proc(*args, **kwargs):
    try:
        actor = args[0]
        port = kwargs.pop("port")
        port.start()
        assert isinstance(port, RecvPort)
        data = port.recv()
        assert np.array_equal(data, prepare_data())
    except Exception as e:
        print("recv error")
        raise e

class Builder():
    def build(self):
        print("this is builder")

class TestShmemChannel(unittest.TestCase):
    mp = MultiProcessing()
    def test_shmemchannel(self):
        self.mp.start()
        size = 5
        predata = prepare_data()
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        print("nbytes:", nbytes)
        name = 'test_shmem_channel'

        shmem_channel = Channel(
            ChannelTransferType.SHMEMCHANNEL,
            size,
            nbytes,
            name)

        send_port = shmem_channel.get_send_port()
        recv_port = shmem_channel.get_recv_port()

        recv_port_fn = partial(recv_proc, port = recv_port)
        send_port_fn = partial(send_proc, port = send_port)

        builder1 = Builder()
        builder2 = Builder()
        self.mp.build_actor(recv_port_fn, builder1)
        self.mp.build_actor(send_port_fn, builder2)

        time.sleep(2)
        self.mp.stop()


if __name__ == "__main__":
    print("start test shardmemory channel")
    unittest.main()
=======
def main():
    data = np.array([12, 24, 36, 48, 60], dtype=np.int32)
    print("Send data: ", data)

    size = 5
    nbytes = nbytes_cal(data.shape, data.dtype)
    name = 'test_shmem_channel'

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
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)

    print(recv_port.recv())
    print(recv_port.recv())
    print(recv_port.recv())
    print(recv_port.recv())
    # print(recv_port.recv())

    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)
    send_port.send(data)

    print(recv_port.recv())
    print(recv_port.recv())

    print("finish test function.")

    send_port.join()
    recv_port.join()


main()
>>>>>>> 8ffebb14fd29b0af6792065cd24354c7b7132ac1
