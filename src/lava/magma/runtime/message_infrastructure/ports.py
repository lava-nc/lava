# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.runtime.message_infrastructure \
    import ChannelQueueSize, CPPSelector
from lava.magma.runtime.message_infrastructure.MessageInfrastructurePywrapper \
    import Channel as CppChannel
from lava.magma.runtime.message_infrastructure.MessageInfrastructurePywrapper \
    import (
        TempChannel,
        support_grpc_channel,
        support_fastdds_channel,
        support_cyclonedds_channel,
        AbstractTransferPort,
        ChannelType,
        RecvPort)

import numpy as np
import typing as ty
import warnings


class Selector(CPPSelector):
    def __init__(self):
        super().__init__()

    def select(self, *args: ty.Tuple[RecvPort, ty.Callable[[], ty.Any]]):
        return super().select(args)


class SendPort(AbstractTransferPort):
    def __init__(self, send_port):
        super().__init__()
        self._cpp_send_port = send_port

    def send(self, data):
        # TODO: Workaround for lava-loihi cpplib, need to change later
        port_type = np.int32 if "LavaCDataType" in str(self.d_type) \
            else self.d_type
        if data.dtype.type != np.str_ and \
                np.dtype(data.dtype).itemsize > np.dtype(port_type).itemsize:
            warnings.warn("Sending data with miss matched dtype,"
                          f"Transfer {data.dtype} to {port_type}")
            data = data.astype(port_type)
        # Use np.copy to handle slices input
        self._cpp_send_port.send(np.copy(data))

    def start(self):
        self._cpp_send_port.start()

    def probe(self):
        return self._cpp_send_port.probe()

    def join(self):
        self._cpp_send_port.join()

    @property
    def name(self):
        return self._cpp_send_port.name

    @property
    def shape(self):
        return self._cpp_send_port.shape

    @property
    def d_type(self):
        return self._cpp_send_port.d_type

    @property
    def size(self):
        return self._cpp_send_port.size

    def get_channel_type(self):
        return self._cpp_send_port.get_channel_type()


if support_grpc_channel():
    from lava.magma.runtime.message_infrastructure. \
        MessageInfrastructurePywrapper \
        import GetRPCChannel as CppRPCChannel

    class GetRPCChannel(CppRPCChannel):

        @property
        def src_port(self):
            return SendPort(super().src_port)

if support_fastdds_channel() or support_cyclonedds_channel():
    from lava.magma.runtime.message_infrastructure. \
        MessageInfrastructurePywrapper \
        import GetDDSChannel as CppDDSChannel

    class GetDDSChannel(CppDDSChannel):
        @property
        def src_port(self):
            return SendPort(super().src_port)


class Channel(CppChannel):

    @property
    def src_port(self):
        return SendPort(super().src_port)


def create_channel(
        message_infrastructure:  \
            "MessageInfrastructureInterface",  # nosec  # noqa
        src_name, dst_name, shape, dtype, size):
    channel_bytes = np.prod(shape) * np.dtype(dtype).itemsize
    return Channel(ChannelType.SHMEMCHANNEL, ChannelQueueSize, channel_bytes,
                   src_name, dst_name, shape, dtype)


def getTempSendPort(addr_path: str):
    tmp_channel = TempChannel(addr_path)
    send_port = tmp_channel.src_port
    return send_port


def getTempRecvPort():
    tmp_channel = TempChannel()
    addr_path = tmp_channel.addr_path
    recv_port = tmp_channel.dst_port
    return addr_path, recv_port
