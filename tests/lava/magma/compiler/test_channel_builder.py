import unittest
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from lava.magma.compiler.builder import ChannelBuilderMp
from lava.magma.compiler.channels.interfaces import Channel
from lava.magma.compiler.utils import PortInitializer
from lava.magma.compiler.channels.pypychannel import (
    PyPyChannel,
    CspSendPort,
    CspRecvPort,
)


# ToDo: (AW) This test does not work for me. Something broken with d_type.
#  SMM does not seem to support numpy types.
class TestChannelBuilder(unittest.TestCase):
    def test_channel_builder(self):
        """Tests Channel Builder creation"""
        smm: SharedMemoryManager = SharedMemoryManager()
        try:
            port_initializer: PortInitializer = PortInitializer(
                name="mock", shape=(1, 2), d_type=np.int32,
                port_type='DOESNOTMATTER', size=64)
            channel_builder: ChannelBuilderMp = ChannelBuilderMp(
                channel_type=PyPyChannel,
                src_port_initializer=port_initializer,
                dst_port_initializer=port_initializer,
                src_process=None,
                dst_process=None,
            )

            smm.start()
            channel: Channel = channel_builder.build(smm)
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
