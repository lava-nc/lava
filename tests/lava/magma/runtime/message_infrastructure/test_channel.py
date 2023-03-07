# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from functools import partial
import time

from lava.magma.runtime.message_infrastructure import (
    PURE_PYTHON_VERSION,
    Channel,
    SendPort,
    RecvPort,
    SupportGRPCChannel,
    SupportFastDDSChannel,
    SupportCycloneDDSChannel
)


def prepare_data():
    return np.random.random_sample((2, 4))


const_data = prepare_data()


def send_proc(*args, **kwargs):
    port = kwargs.pop("port")
    if not isinstance(port, SendPort):
        raise AssertionError()
    port.start()
    port.send(const_data)
    port.join()


def recv_proc(*args, **kwargs):
    port = kwargs.pop("port")
    port.start()
    if not isinstance(port, RecvPort):
        raise AssertionError()
    data = port.recv()
    if not np.array_equal(data, const_data):
        raise AssertionError()
    port.join()


class Builder:
    def build(self):
        pass


def ddschannel_protocol(transfer_type, backend, topic_name):
    from lava.magma.runtime.message_infrastructure import (
        GetDDSChannel,
        DDSBackendType,
        ChannelQueueSize)
    from lava.magma.runtime.message_infrastructure \
        .multiprocessing import MultiProcessing
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

    @unittest.skipIf(PURE_PYTHON_VERSION, "cpp msg lib test")
    def test_shmemchannel(self):
        from lava.magma.runtime.message_infrastructure \
            .MessageInfrastructurePywrapper import ChannelType
        from lava.magma.runtime.message_infrastructure \
            .multiprocessing import MultiProcessing
        from lava.magma.runtime.message_infrastructure \
            import ChannelQueueSize

        mp = MultiProcessing()
        mp.start()
        nbytes = np.prod(const_data.shape) * const_data.dtype.itemsize
        name = 'test_shmem_channel'

        shmem_channel = Channel(
            ChannelType.SHMEMCHANNEL,
            ChannelQueueSize,
            nbytes,
            name,
            name,
            (2, 4),
            const_data.dtype)

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

    @unittest.skipIf(PURE_PYTHON_VERSION, "cpp msg lib test")
    def test_single_process_shmemchannel(self):
        from lava.magma.runtime.message_infrastructure \
            .MessageInfrastructurePywrapper import ChannelType
        from lava.magma.runtime.message_infrastructure \
            import ChannelQueueSize

        predata = prepare_data()
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        name = 'test_single_process_shmem_channel'

        shmem_channel = Channel(
            ChannelType.SHMEMCHANNEL,
            ChannelQueueSize,
            nbytes,
            name,
            name,
            (2, 4),
            const_data.dtype)

        send_port = shmem_channel.src_port
        recv_port = shmem_channel.dst_port

        send_port.start()
        recv_port.start()

        send_port.send(predata)
        resdata = recv_port.recv()

        if not np.array_equal(resdata, predata):
            raise AssertionError()

        self.assertTrue(send_port.shape, (2, 4))
        self.assertTrue(recv_port.d_type, np.int32)

        send_port.join()
        recv_port.join()

    @unittest.skipIf(PURE_PYTHON_VERSION, "cpp msg lib test")
    def test_socketchannel(self):
        from lava.magma.runtime.message_infrastructure \
            .MessageInfrastructurePywrapper import ChannelType
        from lava.magma.runtime.message_infrastructure \
            .multiprocessing import MultiProcessing
        from lava.magma.runtime.message_infrastructure \
            import ChannelQueueSize

        mp = MultiProcessing()
        mp.start()
        nbytes = np.prod(const_data.shape) * const_data.dtype.itemsize
        name = 'test_socket_channel'

        socket_channel = Channel(
            ChannelType.SOCKETCHANNEL,
            ChannelQueueSize,
            nbytes,
            name,
            name,
            (2, 4),
            const_data.dtype)

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

    @unittest.skipIf(PURE_PYTHON_VERSION, "cpp msg lib test")
    def test_single_process_socketchannel(self):
        from lava.magma.runtime.message_infrastructure \
            .MessageInfrastructurePywrapper import ChannelType
        from lava.magma.runtime.message_infrastructure import \
            ChannelQueueSize
        predata = prepare_data()
        nbytes = np.prod(predata.shape) * predata.dtype.itemsize
        name = 'test_single_process_socket_channel'

        socket_channel = Channel(
            ChannelType.SOCKETCHANNEL,
            ChannelQueueSize,
            nbytes,
            name,
            name,
            (2, 4),
            const_data.dtype)

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
        from lava.magma.runtime.message_infrastructure \
            .multiprocessing import MultiProcessing

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
            1)

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
