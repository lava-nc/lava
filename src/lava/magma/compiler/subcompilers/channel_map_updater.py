# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.compiler.channel_map import ChannelMap, Payload, PortPair
from lava.magma.core.process.ports.ports import AbstractSrcPort, AbstractDstPort


class ChannelMapUpdater:
    def __init__(self,
                 channel_map: ChannelMap,
                 payload: ty.Optional[Payload] = None):
        """Offers convenience methods to add entries with a default
        multiplicity into the ChannelMap.

        Parameters
        ----------
        channel_map : ChannelMap
            Channel map that the entries will be entered into.
        payload : Payload, optional
            Data structure that is entered into the channel map for every pair
            of ports.
        """
        self._channel_map = channel_map
        self._payload = payload or Payload(multiplicity=1)

    @property
    def channel_map(self) -> ChannelMap:
        return self._channel_map

    def add_src_ports(self, src_ports: ty.List[AbstractSrcPort]) -> None:
        for src_port in src_ports:
            self.add_src_port(src_port)

    def add_src_port(self, src_port: AbstractSrcPort) -> None:
        for dst_port in src_port.get_dst_ports():
            # If the dst_port is still a source port, then it is
            # a dangling branch and need not be processed.
            if not isinstance(dst_port, AbstractSrcPort):
                self.add_port_pair(src_port, dst_port)

    def add_dst_ports(self, dst_ports: ty.List[AbstractDstPort]) -> None:
        for dst_port in dst_ports:
            self.add_dst_port(dst_port)

    def add_dst_port(self, dst_port: AbstractDstPort) -> None:
        for src_port in dst_port.get_src_ports():
            # If the src_port is still a destination port, then it is
            # a dangling branch and need not be processed.
            if not isinstance(src_port, AbstractDstPort):
                self.add_port_pair(src_port, dst_port)

    def add_port_pair(self,
                      src_port: AbstractSrcPort,
                      dst_port: AbstractDstPort) -> None:
        port_pair = PortPair(src=src_port, dst=dst_port)
        self._channel_map[port_pair] = self._payload

    def add_port_pairs(self, port_pairs) -> None:
        self.channel_map.fromkeys(port_pairs, self._payload)
