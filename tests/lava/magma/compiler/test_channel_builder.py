# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
import unittest

import numpy as np

from lava.magma.compiler.builders.channel_builder import ChannelBuilderMp
from lava.magma.compiler.utils import PortInitializer


from message_infrastructure import (
    ChannelBackend,
    Channel,
    SendPort,
    RecvPort
)


class MockMessageInterface:
    def channel_class(self, channel_type: ChannelBackend) -> ty.Type:
        return ChannelBackend.SHMEMCHANNEL


class TestChannelBuilder(unittest.TestCase):
    def test_channel_builder(self):
        """Tests Channel Builder creation"""
        try:
            port_initializer: PortInitializer = PortInitializer(
                name="mock", shape=(1, 2), d_type=np.int32,
                port_type='DOESNOTMATTER', size=64)
            channel_builder: ChannelBuilderMp = ChannelBuilderMp(
                channel_type=ChannelBackend.SHMEMCHANNEL,
                src_port_initializer=port_initializer,
                dst_port_initializer=port_initializer,
                src_process=None,
                dst_process=None,
            )

            mock = MockMessageInterface()
            channel: Channel = channel_builder.build(mock)
            self.assertIsInstance(channel.src_port, SendPort)
            self.assertIsInstance(channel.dst_port, RecvPort)

            channel.src_port.start()
            channel.dst_port.start()

            expected_data = np.array([[1, 2]], dtype=np.int32)
            channel.src_port.send(expected_data)
            data = channel.dst_port.recv()
            assert np.array_equal(data, expected_data)

            channel.src_port.join()
            channel.dst_port.join()

        finally:
            pass


if __name__ == "__main__":
    unittest.main()
