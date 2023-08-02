# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import unittest

import numpy as np

from lava.magma.compiler.builders.channel_builder import ChannelBuilderMp
from lava.magma.compiler.channels.interfaces import Channel, ChannelType
from lava.magma.compiler.utils import PortInitializer
from lava.magma.compiler.channels.pypychannel import (
    PyPyChannel,
    CspSendPort,
    CspRecvPort,
)
from lava.magma.runtime.message_infrastructure.shared_memory_manager import (
    SharedMemoryManager
)
from lava.magma.compiler.channels.watchdog import NoOPWatchdogManager


class MockMessageInterface:
    def __init__(self, smm):
        self.smm = smm

    def channel_class(self, channel_type: ChannelType) -> ty.Type:
        return PyPyChannel


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
            channel: Channel = channel_builder.build(mock,
                                                     NoOPWatchdogManager())
            assert isinstance(channel, PyPyChannel)
            assert isinstance(channel.src_port, CspSendPort)
            assert isinstance(channel.dst_port, CspRecvPort)

            channel.src_port.start()
            channel.dst_port.start()

            expected_data = np.array([[1, 2]])
            channel.src_port.send(data=expected_data)
            data = channel.dst_port.recv()
            assert np.array_equal(data, expected_data)

            channel.src_port.join()
            channel.dst_port.join()

        finally:
            smm.shutdown()


if __name__ == "__main__":
    unittest.main()
