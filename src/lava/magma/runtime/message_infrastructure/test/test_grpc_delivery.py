# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from functools import partial
import time
from datetime import datetime

from message_infrastructure.multiprocessing import MultiProcessing

from message_infrastructure import (
    ChannelBackend,
    Channel,
    SendPort,
    RecvPort,
    SupportGRPCChannel,
    ChannelQueueSize
)


class Builder:
    def build(self):
        pass


def prepare_data():
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])


def actor_stop(name):
    pass


def target_fn1(*args, **kwargs):
    actor = args[0]
    actor.set_stop_fn(partial(actor_stop, "send"))
    loop = kwargs.pop("loop")
    a1_to_a2 = kwargs.pop("a1_to_a2")
    a2_to_a1 = kwargs.pop("a2_to_a1")
    mp_to_a1 = kwargs.pop("mp_to_a1")
    a1_to_mp = kwargs.pop("a1_to_mp")
    from_mp = mp_to_a1.dst_port
    to_mp = a1_to_mp.src_port
    to_a2 = a1_to_a2.src_port
    from_a2 = a2_to_a1.dst_port
    from_mp.start()
    to_mp.start()
    from_a2.start()
    to_a2.start()
    while loop:
        data = from_mp.recv()
        data[0] = data[0] + 1
        to_a2.send(data)
        data = from_a2.recv()
        data[0] = data[0] + 1
        to_mp.send(data)
        loop -= 1
    from_mp.join()
    to_mp.join()
    from_a2.join()
    to_a2.join()
    actor.status_stopped()


def target_fn2(*args, **kwargs):
    actor = args[0]
    actor.set_stop_fn(partial(actor_stop, "send"))
    loop = kwargs.pop("loop")
    a1_to_a2 = kwargs.pop("a1_to_a2")
    a2_to_a1 = kwargs.pop("a2_to_a1")
    from_a1 = a1_to_a2.dst_port
    to_a1 = a2_to_a1.src_port
    from_a1.start()
    to_a1.start()
    while loop:
        data = from_a1.recv()
        data[0] = data[0] + 1
        to_a1.send(data)
        loop -= 1

    from_a1.join()
    to_a1.join()
    actor.status_stopped()


class TestGrpcDelivery(unittest.TestCase):
    @unittest.skipIf(not SupportGRPCChannel, "Not support grpc channel.")
    def test_grpcchannel(self):
        from message_infrastructure import GetRPCChannel
        mp = MultiProcessing()
        mp.start()
        loop = 10000
        a1_to_a2 = GetRPCChannel(
            '127.13.2.15',
            8003,
            'a1_to_a2',
            'a1_to_a2', 8)
        a2_to_a1 = GetRPCChannel(
            '127.13.2.16',
            8004,
            'a2_to_a1',
            'a2_to_a1', 8)
        mp_to_a1 = GetRPCChannel(
            '127.13.2.17',
            8005,
            'mp_to_a1',
            'mp_to_a1', 8)
        a1_to_mp = GetRPCChannel(
            '127.13.2.18',
            8006,
            'a1_to_mp',
            'a1_to_mp', 8)

        recv_port_fn = partial(target_fn1,
                               loop=loop,
                               mp_to_a1=mp_to_a1,
                               a1_to_mp=a1_to_mp,
                               a1_to_a2=a1_to_a2,
                               a2_to_a1=a2_to_a1)
        send_port_fn = partial(target_fn2,
                               loop=loop,
                               a1_to_a2=a1_to_a2,
                               a2_to_a1=a2_to_a1)
        builder1 = Builder()
        builder2 = Builder()
        mp.build_actor(recv_port_fn, builder1)
        mp.build_actor(send_port_fn, builder2)
        to_a1 = mp_to_a1.src_port
        from_a1 = a1_to_mp.dst_port
        to_a1.start()
        from_a1.start()
        data = prepare_data()
        expect_result = prepare_data()
        expect_result[0] = (1 + 3 * loop)
        loop_start_time = datetime.now()
        while loop:
            to_a1.send(data)
            data = from_a1.recv()
            loop -= 1
        loop_end_time = datetime.now()
        from_a1.join()
        to_a1.join()
        mp.stop(True)
        if not np.array_equal(expect_result, data):
            print("expect: ", expect_result)
            print("result: ", data)
            raise AssertionError()
        print("cpp_grpc_loop_with_cpp_multiprocess timedelta =",
              loop_end_time - loop_start_time)


if __name__ == "__main__":
    unittest.main()
