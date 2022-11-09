# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from functools import partial
import time
from datetime import datetime
from multiprocessing import shared_memory
from multiprocessing import Semaphore
from multiprocessing import Process
from message_infrastructure import GetRPCChannel
from message_infrastructure.multiprocessing import MultiProcessing

from message_infrastructure import (
    ChannelBackend,
    Channel,
    SupportGRPCChannel
)


class process():

    def __init__(self) -> None:
        pass

    def get_status(self):
        return 0


class PyChannel():

    def __init__(self, dtype, size, nbytes, name, *_) -> None:
        self.shm_ = Shm(dtype, size, nbytes, name)
        self.src_port = Port(self.shm_)
        self.dst_port = Port(self.shm_)
        self.dtype_ = dtype
        self.shm_.start()

    def start(self):
        pass

    def join(self):
        pass


class Port():

    def __init__(self, shm) -> None:
        self.shm_ = shm

    def send(self, data):
        return self.shm_.push(data)

    def recv(self):
        return self.shm_.pop()

    def start(self):
        pass

    def join(self):
        pass


class Shm():
    def __init__(self, dtype, size, nbytes, name) -> None:
        self.shm_ = shared_memory.SharedMemory(name=name,
                                               create=True, size=nbytes * size)
        self.nbytes_ = nbytes
        self.size_ = size
        self.sem_ack_ = Semaphore(size)
        self.sem_req_ = Semaphore(0)
        self.sem_ = Semaphore(0)
        self.read_ = 0
        self.write_ = 0
        self.type_ = dtype

    def push(self, data):
        self.sem_ack_.acquire()
        self.sem_.acquire()
        self.shm_.buf[self.write_ * self.nbytes_:
                      ((self.write_ + 1) * self.nbytes_)] = bytearray(data)
        self.write_ = (self.write_ + 1) % self.size_
        self.type_ = data.dtype
        self.sem_.release()
        self.sem_req_.release()

    def pop(self):
        self.sem_req_.acquire()
        self.sem_.acquire()
        result = bytearray(self.shm_.buf[self.read_ * self.nbytes_:
                           ((self.read_ + 1) * self.nbytes_)])
        self.read_ = (self.read_ + 1) % self.size_
        self.sem_.release()
        self.sem_ack_.release()
        return np.frombuffer(result, self.type_)

    def start(self):
        self.sem_.release()

    def __del__(self):
        self.shm_.close()
        self.shm_.unlink()


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
        time.sleep(0.0001)
    from_mp.join()
    to_a2.join()
    from_a2.join()
    to_mp.join()


def bound_target_a2(loop, a1_to_a2, a2_to_a1, this, builder):
    from_a1 = a1_to_a2.dst_port
    from_a1.start()
    to_a1 = a2_to_a1.src_port
    to_a1.start()
    while loop > 0 and this.get_status() == 0:
        loop = loop - 1
        data = from_a1.recv()
        data[0] = data[0] + 1
        to_a1.send(data)

    while this.get_status() == 0:
        time.sleep(0.0001)
    from_a1.join()
    to_a1.join()


def prepare_data():
    arr1 = np.array([1] * 9990)
    arr2 = np.array([1, 2, 3, 4, 5,
                    6, 7, 8, 9, 0])
    return np.concatenate((arr2, arr1))


class TestShmDelivery(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.loop_ = 100000

    # @unittest.skip("(cpp_shm_loop_with_cpp_multiprocess")
    def test_cpp_shm_loop_with_cpp_multiprocess(self):
        loop = self.loop_
        mp = MultiProcessing()
        mp.start()
        predata = prepare_data()
        queue_size = 1
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

        mp.build_actor(target_a1, builder)  # actor1
        mp.build_actor(target_a2, builder)  # actor2

        to_a1 = mp_to_a1.src_port
        from_a1 = a1_to_mp.dst_port

        to_a1.start()
        from_a1.start()

        expect_result = np.copy(predata)
        expect_result[0] = (1 + 3 * loop)
        loop_start = datetime.now()
        while loop > 0:
            loop = loop - 1
            to_a1.send(predata)
            predata = from_a1.recv()
        loop_end = datetime.now()
        print("cpp_shm_loop_with_cpp_multiprocess result = ", predata[0])
        if not np.array_equal(expect_result, predata):
            print("expect: ", expect_result)
            print("result: ", predata)
            raise AssertionError()

        to_a1.join()
        from_a1.join()
        mp.stop(True)
        print("cpp_shm_loop_with_cpp_multiprocess timedelta =",
              loop_end - loop_start)

    # @unittest.skip("cpp_skt_loop_with_cpp_multiprocess")
    def test_cpp_skt_loop_with_cpp_multiprocess(self):
        loop = self.loop_
        mp = MultiProcessing()
        mp.start()
        predata = prepare_data()
        queue_size = 2
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        mp_to_a1 = Channel(
            ChannelBackend.SOCKETCHANNEL,
            queue_size,
            nbytes,
            "mp_to_a1",
            "mp_to_a1")
        a1_to_a2 = Channel(
            ChannelBackend.SOCKETCHANNEL,
            queue_size,
            nbytes,
            "a1_to_a2",
            "a1_to_a2")
        a2_to_a1 = Channel(
            ChannelBackend.SOCKETCHANNEL,
            queue_size,
            nbytes,
            "a2_to_a1",
            "a2_to_a1")
        a1_to_mp = Channel(
            ChannelBackend.SOCKETCHANNEL,
            queue_size,
            nbytes,
            "a1_to_mp",
            "a1_to_mp")

        target_a1 = partial(bound_target_a1, loop, mp_to_a1,
                            a1_to_a2, a2_to_a1, a1_to_mp)
        target_a2 = partial(bound_target_a2, loop, a1_to_a2, a2_to_a1)

        builder = Builder()

        mp.build_actor(target_a1, builder)  # actor1
        mp.build_actor(target_a2, builder)  # actor2

        to_a1 = mp_to_a1.src_port
        from_a1 = a1_to_mp.dst_port

        to_a1.start()
        from_a1.start()

        expect_result = np.copy(predata)
        expect_result[0] = (1 + 3 * loop)
        loop_start = datetime.now()
        while loop > 0:
            loop = loop - 1
            to_a1.send(predata)
            predata = from_a1.recv()
        loop_end = datetime.now()
        print("cpp_skt_loop_with_cpp_multiprocess result = ", predata[0])
        if not np.array_equal(expect_result, predata):
            print("expect: ", expect_result)
            print("result: ", predata)
            raise AssertionError()

        to_a1.join()
        from_a1.join()
        mp.stop(True)
        print("cpp_skt_loop_with_cpp_multiprocess timedelta =",
              loop_end - loop_start)

    # @unittest.skip("py_shm_loop_with_cpp_multiprocess")
    def test_py_shm_loop_with_cpp_multiprocess(self):
        loop = self.loop_

        mp = MultiProcessing()
        mp.start()

        predata = prepare_data()
        queue_size = 1
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        mp_to_a1 = PyChannel(
            predata.dtype,
            queue_size,
            nbytes,
            "mp_to_a1",
            "mp_to_a1")
        a1_to_a2 = PyChannel(
            predata.dtype,
            queue_size,
            nbytes,
            "a1_to_a2",
            "a1_to_a2")
        a2_to_a1 = PyChannel(
            predata.dtype,
            queue_size,
            nbytes,
            "a2_to_a1",
            "a2_to_a1")
        a1_to_mp = PyChannel(
            predata.dtype,
            queue_size,
            nbytes,
            "a1_to_mp",
            "a1_to_mp")

        builder = Builder()

        target_a1 = partial(bound_target_a1, loop, mp_to_a1,
                            a1_to_a2, a2_to_a1, a1_to_mp)
        target_a2 = partial(bound_target_a2, loop, a1_to_a2, a2_to_a1)

        mp.build_actor(target_a1, builder)  # actor1
        mp.build_actor(target_a2, builder)  # actor2

        to_a1 = mp_to_a1.src_port
        from_a1 = a1_to_mp.dst_port

        to_a1.start()
        from_a1.start()

        expect_result = np.copy(predata)
        expect_result[0] = (1 + 3 * loop)

        loop_start = datetime.now()
        while loop > 0:
            loop = loop - 1
            to_a1.send(predata)
            predata = from_a1.recv()
        loop_end = datetime.now()
        print("py_shm_loop_with_cpp_multiprocess result = ", predata[0])
        if not np.array_equal(expect_result, predata):
            print("expect: ", expect_result)
            print("result: ", predata)
            raise AssertionError()

        to_a1.join()
        from_a1.join()
        mp.stop(True)
        print("py_shm_loop_with_cpp_multiprocess timedelta =",
              loop_end - loop_start)

    # @unittest.skip("test_py_shm_loop_with_py_multiprocess")
    def test_py_shm_loop_with_py_multiprocess(self):
        loop = self.loop_

        predata = prepare_data()
        queue_size = 1
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        mp_to_a1 = PyChannel(
            predata.dtype,
            queue_size,
            nbytes,
            "mp_to_a1",
            "mp_to_a1")
        a1_to_a2 = PyChannel(
            predata.dtype,
            queue_size,
            nbytes,
            "a1_to_a2",
            "a1_to_a2")
        a2_to_a1 = PyChannel(
            predata.dtype,
            queue_size,
            nbytes,
            "a2_to_a1",
            "a2_to_a1")
        a1_to_mp = PyChannel(
            predata.dtype,
            queue_size,
            nbytes,
            "a1_to_mp",
            "a1_to_mp")

        builder = Builder()

        target_a1 = partial(bound_target_a1, loop, mp_to_a1,
                            a1_to_a2, a2_to_a1, a1_to_mp, process(), builder)
        target_a2 = partial(bound_target_a2, loop, a1_to_a2, a2_to_a1,
                            process(), builder)

        a1 = Process(target=target_a1)
        a2 = Process(target=target_a2)
        a1.start()
        a2.start()

        to_a1 = mp_to_a1.src_port
        from_a1 = a1_to_mp.dst_port

        to_a1.start()
        from_a1.start()

        expect_result = np.copy(predata)
        expect_result[0] = (1 + 3 * loop)

        loop_start = datetime.now()
        while loop > 0:
            loop = loop - 1
            to_a1.send(predata)
            predata = from_a1.recv()
        loop_end = datetime.now()
        print("py_shm_loop_with_py_multiprocess result = ", predata[0])
        if not np.array_equal(expect_result, predata):
            print("expect: ", expect_result)
            print("result: ", predata)
            raise AssertionError()

        to_a1.join()
        from_a1.join()
        a1.terminate()
        a2.terminate()
        a1.join()
        a2.join()
        print("py_shm_loop_with_py_multiprocess timedelta =",
              loop_end - loop_start)

    @unittest.skip("cpp_grpc_loop_with_cpp_multiprocess")
    def test_grpcchannel(self):
        mp = MultiProcessing()
        mp.start()
        loop = self.loop_
        a1_to_a2 = GetRPCChannel(
            '127.13.2.11',
            8001,
            'a1_to_a2',
            'a1_to_a2', 8)
        a2_to_a1 = GetRPCChannel(
            '127.13.2.12',
            8002,
            'a2_to_a1',
            'a2_to_a1', 8)
        mp_to_a1 = GetRPCChannel(
            '127.13.2.13',
            8003,
            'mp_to_a1',
            'mp_to_a1', 8)
        a1_to_mp = GetRPCChannel(
            '127.13.2.14',
            8004,
            'a1_to_mp',
            'a1_to_mp', 8)

        recv_port_fn = partial(bound_target_a1, loop, mp_to_a1,
                               a1_to_a2, a2_to_a1, a1_to_mp)
        send_port_fn = partial(bound_target_a2, loop, a1_to_a2, a2_to_a1)

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
        print("cpp_grpc_loop_with_cpp_multiprocess result = ", data[0])
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


if __name__ == '__main__':
    unittest.main()
