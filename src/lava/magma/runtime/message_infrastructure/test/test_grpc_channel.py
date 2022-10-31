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
    SupportGRPCChannel,
    ChannelQueueSize
)


def prepare_data():
    return np.random.random_sample((2, 4))


const_data_x = prepare_data()


def actor_stop(name):
    pass


def send_proc(*args, **kwargs):
    actor = args[0]
    actor.set_stop_fn(partial(actor_stop, "send"))
    port = kwargs.pop("port")
    if not isinstance(port, SendPort):
        raise AssertionError()
    port.start()
    print("send data: ", const_data_x)
    port.send(const_data_x)
    print("send data done: ", const_data_x)
    port.join()
    actor.status_stopped()


def recv_proc(*args, **kwargs):
    actor = args[0]
    actor.set_stop_fn(partial(actor_stop, "recv"))
    port = kwargs.pop("port")
    port.start()
    if not isinstance(port, RecvPort):
        raise AssertionError()
    print("before recv data: ")
    data = port.recv()
    print("recv data: ", data)
    if not np.array_equal(data, const_data_x):
        raise AssertionError()
    port.join()
    actor.status_stopped()


class Builder:
    def build(self):
        pass


class TestChannel(unittest.TestCase):

    # @unittest.skipIf(not SupportGRPCChannel, "Not support grpc channel.")
    # @unittest.skip("")
    def test_grpcchannel(self):
        from message_infrastructure import GetRPCChannel
        mp = MultiProcessing()
        mp.start()
        name = 'test_grpc_channel'
        url = '127.13.2.9'
        port = 8003
        grpc_channel = GetRPCChannel(
            url,
            port,
            name,
            name,
            ChannelQueueSize)

        send_port = grpc_channel.src_port
        recv_port = grpc_channel.dst_port

        recv_port_fn = partial(recv_proc, port=recv_port)
        send_port_fn = partial(send_proc, port=send_port)

        builder1 = Builder()
        builder2 = Builder()
        mp.build_actor(recv_port_fn, builder1)
        mp.build_actor(send_port_fn, builder2)

        time.sleep(0.1)
        mp.stop(True)


if __name__ == "__main__":
    unittest.main()
