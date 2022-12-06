# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from functools import partial
import time

from lava.magma.runtime.message_infrastructure.multiprocessing import MultiProcessing

from lava.magma.runtime.message_infrastructure import (
    ChannelBackend,
    Channel,
    SendPort,
    RecvPort,
    SupportGRPCChannel,
    SupportFastDDSChannel,
    SupportCycloneDDSChannel,
    ChannelQueueSize
)


def prepare_data():
    return np.random.random_sample((2, 4))


const_data = prepare_data()


def actor_stop(name):
    pass


def send_proc(*args, **kwargs):
    actor = args[0]
    actor.set_stop_fn(partial(actor_stop, "send"))
    port = kwargs.pop("port")
    if not isinstance(port, SendPort):
        raise AssertionError()
    port.start()
    port.send(const_data)
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
    if not np.array_equal(data, const_data):
        raise AssertionError()
    port.join()
    actor.status_stopped()


class Builder:
    def build(self):
        pass


def ddschannel_protocol(transfer_type, backend, topic_name):
    from lava.magma.runtime.message_infrastructure import GetDDSChannel
    from lava.magma.runtime.message_infrastructure import DDSTransportType
    from lava.magma.runtime.message_infrastructure import DDSBackendType
    mp = MultiProcessing()
    mp.start()
    dds_channel = GetDDSChannel(
        topic_name,
        transfer_type,
        backend,
        ChannelQueueSize)

    send_port = dds_channel.src_port
    recv_port = dds_channel.dst_port

    recv_port_fn = partial(recv_proc, port=recv_port)
    send_port_fn = partial(send_proc, port=send_port)

    builder1 = Builder()
    builder2 = Builder()
    mp.build_actor(recv_port_fn, builder1)
    mp.build_actor(send_port_fn, builder2)

    time.sleep(0.1)
    mp.stop()
    mp.cleanup(True)


class TestChannel(unittest.TestCase):

    def test_shmemchannel(self):
        mp = MultiProcessing()
        mp.start()
        nbytes = np.prod(const_data.shape) * const_data.dtype.itemsize
        name = 'test_shmem_channel'

        shmem_channel = Channel(
            ChannelBackend.SHMEMCHANNEL,
            ChannelQueueSize,
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

        time.sleep(0.1)
        mp.stop()
        mp.cleanup(True)

    def test_single_process_shmemchannel(self):
        predata = prepare_data()
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        name = 'test_single_process_shmem_channel'

        shmem_channel = Channel(
            ChannelBackend.SHMEMCHANNEL,
            ChannelQueueSize,
            nbytes,
            name,
            name)

        send_port = shmem_channel.src_port
        recv_port = shmem_channel.dst_port

        send_port.start()
        recv_port.start()

        send_port.send(predata)
        resdata = recv_port.recv()

        if not np.array_equal(resdata, predata):
            raise AssertionError()

        send_port.join()
        recv_port.join()

    def test_socketchannel(self):
        mp = MultiProcessing()
        mp.start()
        nbytes = np.prod(const_data.shape) * const_data.dtype.itemsize
        name = 'test_socket_channel'

        socket_channel = Channel(
            ChannelBackend.SOCKETCHANNEL,
            ChannelQueueSize,
            nbytes,
            name,
            name)

        send_port = socket_channel.src_port
        recv_port = socket_channel.dst_port

        recv_port_fn = partial(recv_proc, port=recv_port)
        send_port_fn = partial(send_proc, port=send_port)

        builder1 = Builder()
        builder2 = Builder()
        mp.build_actor(recv_port_fn, builder1)
        mp.build_actor(send_port_fn, builder2)

        time.sleep(0.1)
        mp.stop()
        mp.cleanup(True)

    def test_single_process_socketchannel(self):
        predata = prepare_data()
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        name = 'test_single_process_socket_channel'

        socket_channel = Channel(
            ChannelBackend.SOCKETCHANNEL,
            ChannelQueueSize,
            nbytes,
            name,
            name)

        send_port = socket_channel.src_port
        recv_port = socket_channel.dst_port

        send_port.start()
        recv_port.start()

        send_port.send(predata)
        resdata = recv_port.recv()

        if not np.array_equal(resdata, predata):
            raise AssertionError()

        send_port.join()
        recv_port.join()

    @unittest.skipIf(not SupportGRPCChannel, "Not support grpc channel.")
    def test_grpcchannel(self):
        from lava.magma.runtime.message_infrastructure import GetRPCChannel
        mp = MultiProcessing()
        mp.start()
        name = 'test_grpc_channel'
        url = '127.13.2.11'
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
        mp.stop()
        mp.cleanup(True)

    @unittest.skipIf(not SupportFastDDSChannel, "Not support fastdds channel.")
    def test_fastdds_channel_shm(self):
        from lava.magma.runtime.message_infrastructure import DDSTransportType
        from lava.magma.runtime.message_infrastructure import DDSBackendType
        ddschannel_protocol(DDSTransportType.DDSSHM,
                            DDSBackendType.FASTDDSBackend,
                            "test_fastdds_channel_shm")

    @unittest.skipIf(not SupportFastDDSChannel, "Not support fastdds channel.")
    def test_fastdds_channel_udpv4(self):
        from lava.magma.runtime.message_infrastructure import DDSTransportType
        from lava.magma.runtime.message_infrastructure import DDSBackendType
        ddschannel_protocol(DDSTransportType.DDSUDPv4,
                            DDSBackendType.FASTDDSBackend,
                            "test_fastdds_channel_udpv4")

    @unittest.skipIf(not SupportCycloneDDSChannel,
                     "Not support cyclonedds channel.")
    def test_cyclonedds_channel_shm(self):
        from lava.magma.runtime.message_infrastructure import DDSTransportType
        from lava.magma.runtime.message_infrastructure import DDSBackendType
        ddschannel_protocol(DDSTransportType.DDSSHM,
                            DDSBackendType.CycloneDDSBackend,
                            "test_cyclonedds_shm")

    @unittest.skipIf(not SupportCycloneDDSChannel,
                     "Not support cyclonedds channel.")
    def test_cyclonedds_channel_udpv4(self):
        from lava.magma.runtime.message_infrastructure import DDSTransportType
        from lava.magma.runtime.message_infrastructure import DDSBackendType
        ddschannel_protocol(DDSTransportType.DDSUDPv4,
                            DDSBackendType.CycloneDDSBackend,
                            "test_cyclonedds_udpv4")


if __name__ == "__main__":
    unittest.main()
