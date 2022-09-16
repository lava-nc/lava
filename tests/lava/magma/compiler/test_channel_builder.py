# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
import unittest

import numpy as np

from lava.magma.compiler.builders.channel_builder import ChannelBuilderMp
from lava.magma.compiler.utils import PortInitializer

from message_infrastructure import (
    ChannelTransferType,
    Channel,
    SendPort,
    RecvPort
)


class MockMessageInterface:
    def channel_class(self, channel_type: ChannelTransferType) -> ty.Type:
        return ChannelTransferType.SHMEMCHANNEL


class TestChannelBuilder(unittest.TestCase):
    def test_channel_builder(self):
        """Tests Channel Builder creation"""
        try:
            port_initializer: PortInitializer = PortInitializer(
                name="mock", shape=(5), d_type=np.int32,
                port_type='DOESNOTMATTER', size=5)
            channel_builder: ChannelBuilderMp = ChannelBuilderMp(
                channel_type=ChannelTransferType.SHMEMCHANNEL,
                src_port_initializer=port_initializer,
                dst_port_initializer=port_initializer,
                src_process=None,
                dst_process=None,
            )

            mock = MockMessageInterface()
            channel: Channel = channel_builder.build(mock)
            src_port = channel.get_send_port()
            dst_port = channel.get_recv_port()
            # assert isinstance(channel, ShmemChannel)
            assert isinstance(src_port, SendPort)
            assert isinstance(dst_port, RecvPort)

            src_port.start()
            dst_port.start()

            expected_data = np.array([12,34,36,48,60], dtype = np.int32)
            src_port.send(expected_data)
            data = dst_port.recv()
            assert np.array_equal(data, expected_data)
            print("final OK")

            src_port.join()
            dst_port.join()

        finally:
            print("End Test")


if __name__ == "__main__":
    print("start test channel builder")
    unittest.main()
