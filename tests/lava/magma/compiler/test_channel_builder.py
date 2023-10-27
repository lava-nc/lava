# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import unittest

import numpy as np
from multiprocessing.managers import SharedMemoryManager
from lava.magma.compiler.builders.channel_builder import ChannelBuilderMp
from lava.magma.compiler.utils import PortInitializer

from lava.magma.runtime.message_infrastructure import (
    Channel,
    SendPort,
    RecvPort,
    create_channel
)
from lava.magma.runtime.message_infrastructure.interfaces import ChannelType
from lava.magma.runtime.message_infrastructure.shared_memory_manager import (
    SharedMemoryManager
)
from lava.magma.runtime.message_infrastructure.watchdog import \
    NoOPWatchdogManager


class MockMessageInterface:
    def __init__(self, smm):
        self.smm = smm

    def channel(self, channel_type: ChannelType, src_name, dst_name,
                shape, dtype, size) -> Channel:
        return create_channel(self, src_name, dst_name, shape, dtype, size)


class TestChannelBuilder(unittest.TestCase):
    def test_channel_builder(self):
        """Tests Channel Builder creation"""
        smm = SharedMemoryManager()
        try:
            port_initializer: PortInitializer = PortInitializer(
                name="mock", shape=(1, 2), d_type=np.int32,
                port_type='DOESNOTMATTER', size=64)
            channel_builder: ChannelBuilderMp = ChannelBuilderMp(
                channel_type=ChannelType.PyPy,
                src_port_initializer=port_initializer,
                dst_port_initializer=port_initializer,
                src_process=None,
                dst_process=None,
            )
            smm.start()
            mock = MockMessageInterface(smm)
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
            smm.shutdown()


if __name__ == "__main__":
    unittest.main()
