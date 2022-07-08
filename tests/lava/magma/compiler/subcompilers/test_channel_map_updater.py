# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.compiler.channel_map import ChannelMap, Payload, PortPair
from lava.magma.compiler.subcompilers.channel_map_updater import \
    ChannelMapUpdater
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess


class Process(AbstractProcess):
    def __init__(self):
        super().__init__()
        self.inp = InPort(shape=(1,))
        self.inp2 = InPort(shape=(2,))
        self.out = OutPort(shape=(1,))
        self.out2 = OutPort(shape=(2,))


class TestChannelMapUpdater(unittest.TestCase):
    def setUp(self) -> None:
        self.channel_map = ChannelMap()
        self.payload = Payload(multiplicity=5)
        self.updater = ChannelMapUpdater(self.channel_map, self.payload)

    def test_init_default(self) -> None:
        channel_map = ChannelMap()
        updater = ChannelMapUpdater(channel_map)
        self.assertEqual(updater.channel_map, channel_map)
        self.assertEqual(updater._payload.multiplicity, 1)

    def test_init_with_multiplicity(self) -> None:
        self.assertEqual(self.updater.channel_map,
                         self.channel_map)
        self.assertEqual(self.updater._payload.multiplicity, 5)

    def test_add_port_pair(self) -> None:
        process = Process()
        self.updater.add_port_pair(process.out, process.inp)

        self.assertEqual(len(self.updater.channel_map), 1)
        port_pair = PortPair(src=process.out, dst=process.inp)
        self.assertEqual(self.updater.channel_map[port_pair], self.payload)

    def test_add_dst_port(self) -> None:
        source = Process()
        destination = Process()
        source.out.connect(destination.inp)
        self.updater.add_dst_port(destination.inp)

        self.assertEqual(len(self.updater.channel_map), 1)
        port_pair = PortPair(src=source.out, dst=destination.inp)
        self.assertEqual(self.updater.channel_map[port_pair], self.payload)

    def test_add_src_port(self) -> None:
        source = Process()
        destination = Process()
        source.out.connect(destination.inp)
        self.updater.add_src_port(source.out)

        self.assertEqual(len(self.updater.channel_map), 1)
        port_pair = PortPair(src=source.out, dst=destination.inp)
        self.assertEqual(self.updater.channel_map[port_pair], self.payload)

    def test_add_dst_ports(self) -> None:
        source = Process()
        destination = Process()
        source.out.connect(destination.inp)
        source.out2.connect(destination.inp2)
        self.updater.add_dst_ports([destination.inp, destination.inp2])

        self.assertEqual(len(self.updater.channel_map), 2)
        port_pair1 = PortPair(src=source.out, dst=destination.inp)
        self.assertEqual(self.updater.channel_map[port_pair1], self.payload)
        port_pair2 = PortPair(src=source.out2, dst=destination.inp2)
        self.assertEqual(self.updater.channel_map[port_pair2], self.payload)

    def test_add_src_ports(self) -> None:
        source = Process()
        destination = Process()
        source.out.connect(destination.inp)
        source.out2.connect(destination.inp2)
        self.updater.add_src_ports([source.out, source.out2])

        self.assertEqual(len(self.updater.channel_map), 2)
        port_pair1 = PortPair(src=source.out, dst=destination.inp)
        self.assertEqual(self.updater.channel_map[port_pair1], self.payload)
        port_pair2 = PortPair(src=source.out2, dst=destination.inp2)
        self.assertEqual(self.updater.channel_map[port_pair2], self.payload)


if __name__ == '__main__':
    unittest.main()
