# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from traceback import print_tb
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


def prepare_data():
    return np.random.random_sample((2, 4))


def actor_stop(name):
    pass


def send_proc(*args, **kwargs):
    actor = args[0]
    actor.set_stop_fn(partial(actor_stop, "send"))
    port = kwargs.pop("port")
    if not isinstance(port, SendPort):
        raise AssertionError()
    port.start()
    senddata = prepare_data()
    port.send(senddata)
    print("send data = = = ",senddata)
    print("send data ok")
    port.join()
    actor.status_stopped()


def recv_proc(*args, **kwargs):
    actor = args[0]
    actor.set_stop_fn(partial(actor_stop, "recv"))
    port = kwargs.pop("port")
    port.start()
    if not isinstance(port, RecvPort):
        raise AssertionError()
    data = port.recv()
    print("recvdata = = == = =",data)
    if not np.array_equal(data, prepare_data()):
        raise AssertionError()
    print("recv data ok")
    port.join()
    actor.status_stopped()


class Builder:
    def build(self):
        pass


class TestChannel(unittest.TestCase):

    def test_shmemchannel(self):
        print("test_shmemchannel======")
        mp = MultiProcessing()
        mp.start()
        size = 5
        predata = prepare_data()
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        name = 'test_shmem_channel'

        shmem_channel = Channel(
            ChannelBackend.RPCCHANNEL,
            size,
            nbytes,
            name,
            name)

        send_port = shmem_channel.src_port
        recv_port = shmem_channel.dst_port

        recv_port_fn = partial(recv_proc, port=recv_port)
        send_port_fn = partial(send_proc, port=send_port)

        builder1 = Builder()
        builder2 = Builder()
        mp.build_actor(recv_port_fn, builder1)
        mp.build_actor(send_port_fn, builder2)

        time.sleep()
        mp.stop(True)
        print("test_shmemchannel==============")

    def test_single_process_socketchannel(self):

        size = 5
        predata = prepare_data()
        predata2 = prepare_data()
        print(predata)
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        name = 'test_single_process_socket_channel'

        socket_channel = Channel(
            ChannelBackend.RPCCHANNEL,
            size,
            nbytes,
            name,
            name)

        send_port = socket_channel.src_port
        recv_port = socket_channel.dst_port

        send_port.start()
        print("send start OK")
        recv_port.start()
        print("recv start OK")
        send_port.send(predata)
        send_port.send(predata2)
        send_port.send(predata)
        send_port.send(predata)
        print("send ok")
        resdata = recv_port.recv()
        resdata2 = recv_port.recv()
        print("recv ok")
        print(resdata)
        print(resdata2)
        if not np.array_equal(resdata, predata):
            raise AssertionError()
        if not np.array_equal(resdata2, predata2):
            raise AssertionError()     
        ######
        send_port.join()
        print("send join ok")
        recv_port.join()
        print("recv join ok")


if __name__ == "__main__":
    unittest.main()
