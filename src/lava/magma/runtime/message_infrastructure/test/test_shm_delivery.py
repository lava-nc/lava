# Copyright (C) 2021 Intel Corporation
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
)


class Builder():
    def build(self, i):
        pass


def bound_target_a1(loop, mp_to_a1, a1_to_a2,
                    a2_to_a1, a1_to_mp, this, builder):
    from_mp = mp_to_a1.dst_port
    from_mp.start()
    to_a2 = a1_to_a2.src_port
    to_a2.start()
    from_a2 = a2_to_a1.dst_port
    from_a2.start()
    to_mp = a1_to_mp.src_port
    to_mp.start()
    while loop > 0 and this.get_status() == 0:
        loop = loop - 1
        data = from_mp.recv()
        data[0] = data[0] + 1
        to_a2.send(data)
        data = from_a2.recv()
        data[0] = data[0] + 1
        to_mp.send(data)

    while this.get_status() == 0:
        time.sleep(1)
    from_mp.join()
    to_a2.join()
    from_a2.join()
    to_mp.join()


def bound_target_a2(loop, a1_to_a2, a2_to_a1, this, builder):
    from_a1 = a1_to_a2.dst_port
    from_a1.start()
    to_a1 = a2_to_a1.src_port
    to_a1.start()
    print("actor2 status: ", end="")
    print(this.get_status())
    while loop > 0 and this.get_status() == 0:
        loop = loop - 1
        data = from_a1.recv()
        data[0] = data[0] + 1
        to_a1.send(data)

    while this.get_status() == 0:
        time.sleep(0.5)
    from_a1.join()
    to_a1.join()


def prepare_data():
    return np.array([1])


class TestShmDelivery(unittest.TestCase):

    def test_shm_loop(this):
        loop = 10000
        mp = MultiProcessing()
        mp.start()
        predata = prepare_data()
        queue_size = 2
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        mp_to_a1 = Channel(
            ChannelBackend.SHMEMCHANNEL,
            queue_size,
            nbytes,
            "mp_to_a1",
            "mp_to_a1")
        a1_to_a2 = Channel(
            ChannelBackend.SHMEMCHANNEL,
            queue_size,
            nbytes,
            "a1_to_a2",
            "a1_to_a2")
        a2_to_a1 = Channel(
            ChannelBackend.SHMEMCHANNEL,
            queue_size,
            nbytes,
            "a2_to_a1",
            "a2_to_a1")
        a1_to_mp = Channel(
            ChannelBackend.SHMEMCHANNEL,
            queue_size,
            nbytes,
            "a1_to_mp",
            "a1_to_mp")

        target_a1 = partial(bound_target_a1, loop, mp_to_a1,
                            a1_to_a2, a2_to_a1, a1_to_mp)
        target_a2 = partial(bound_target_a2, loop, a1_to_a2, a2_to_a1)

        builder = Builder()

        _ = mp.build_actor(target_a1, builder)  # actor1
        _ = mp.build_actor(target_a2, builder)  # actor2

        to_a1 = mp_to_a1.src_port
        from_a1 = a1_to_mp.dst_port

        to_a1.start()
        from_a1.start()

        expect_result = np.array([1 + 3 * loop])
        while loop > 0:
            loop = loop - 1
            to_a1.send(predata)
            predata = from_a1.recv()

        print("result: ", end="")
        print(predata)
        if not np.array_equal(expect_result, predata):
            raise AssertionError()

        to_a1.join()
        from_a1.join()
        mp.stop(True)


if __name__ == '__main__':
    unittest.main()
