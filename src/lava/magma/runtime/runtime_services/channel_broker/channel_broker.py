# Copyright (C) 2022-23 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import threading
from abc import ABC
import logging

import numpy as np
import typing as ty

from lava.magma.compiler.channels.interfaces import AbstractCspPort
from lava.magma.compiler.channels.pypychannel import CspSelector, PyPyChannel
from lava.magma.runtime.message_infrastructure.shared_memory_manager import (
    SharedMemoryManager,
)

try:
    from nxcore.arch.base.nxboard import NxBoard
    from nxcore.graph.channel import Channel
    from nxcore.graph.processes.embedded.embedded_snip import EmbeddedSnip
except ImportError:
    class NxBoard:
        pass

    class Channel:
        pass

    class Phase:
        pass

    class EmbeddedSnip:
        pass

try:
    from lava.magma.core.model.c.ports import AbstractCPort, CInPort, COutPort
except ImportError:
    class AbstractCPort:
        pass

    class CInPort:
        pass

    class COutPort:
        pass


class AbstractChannelBroker(ABC):
    """Abstract ChannelBroker."""

    def __init__(self,
                 *args,
                 **kwargs):
        """Initialize the abstract ChannelBroker with logging."""
        self.log = logging.getLogger(__name__)
        self.log.setLevel(kwargs.get("loglevel", logging.WARNING))


def generate_channel_name(prefix: str,
                          port_idx: int,
                          csp_port: AbstractCspPort,
                          c_builder_idx: int) -> str:
    return f"{prefix}{str(port_idx)}_{str(csp_port.name)}_{str(c_builder_idx)}"


class ChannelBroker(AbstractChannelBroker):
    """ChannelBroker for NxSdkRuntimeService.

    ChannelBroker handles communication between NxSdkRuntimeService,
    Lava Processes and a Loihi board. It uses the NxCore
    board object and creates GRPC Channels for each port.
    The ChannelBroker sends messages over the GRPC
    channels and services requests by the NxSdkRuntimeService.
    The NxSdkRuntimeService intercepts CPort and NcPort
    messages from the Runtime and brokers the communication between
    runtime and Loihi Boards.
    """

    def __init__(self,
                 board: NxBoard,
                 compile_config: ty.Optional[ty.Dict[str, ty.Any]] = None,
                 *args,
                 **kwargs,):
        """Initialize ChannelBroker with NxBoard.

        Parameters
        ----------
        board : NxBoard
            Board object to use when creating channels.
        *args : any
        **kwargs : any
        """
        super().__init__(*args, **kwargs)
        self.board = board
        self._compile_config = compile_config
        self.has_started: bool = False
        # Need to poll for CInPorts
        self.c_inports_to_poll: ty.Dict[CInPort, Channel] = {}

        # Need to pill for COutPorts
        self.c_outports_to_poll: ty.Dict[Channel, COutPort] = {}

        self.smm: SharedMemoryManager = SharedMemoryManager()
        self.mgmt_channel: ty.Optional[PyPyChannel] = None
        self.grpc_stopping_event: ty.Optional[threading.Event] = None
        self.port_poller: ty.Optional[threading.Thread] = None
        self.grpc_poller: ty.Optional[threading.Thread] = None

    def run(self):
        """Start the polling threads"""
        if not self.has_started:
            self.smm.start()
            self.mgmt_channel = PyPyChannel(
                message_infrastructure=self,
                src_name="mgmt_channel",
                dst_name="mgmt_channel",
                shape=(1,),
                dtype=np.int32,
                size=1)
            self.mgmt_channel.src_port.start()
            self.mgmt_channel.dst_port.start()
            self.port_poller = threading.Thread(target=self.poll_c_inports)
            self.grpc_stopping_event = threading.Event()
            self.grpc_poller = threading.Thread(target=self.poll_c_outports)
            self.port_poller.start()
            self.grpc_poller.start()
            self.has_started = True

    def join(self):
        if self.has_started:
            self.grpc_stopping_event.set()
            self.mgmt_channel.src_port.send(np.array([0]))
            self.grpc_poller.join()
            self.port_poller.join()
            self.mgmt_channel.dst_port.join()
            self.mgmt_channel.src_port.join()
            self.has_started = False
            self.smm.shutdown()

    def poll_c_inports(self):
        """Retrieves request from processes and brokers communication
        between Cproc Models in Python and Cproc Models in C.
        After sending requests to the GRPC channel the process
        is informed about completion.
        """
        selector = CspSelector()

        while True:
            # Need to poll both GRPC and CSP ports for messages
            # Service read write requests here
            channel_actions = []
            for cport, channel in self.c_inports_to_poll.items():
                result = (cport, channel)
                channel_actions.append((cport.csp_ports[0],
                                        (lambda y: (lambda: y))(result)))

            channel_actions.append((self.mgmt_channel.dst_port,
                                    lambda: ('stop', None)))
            action, channel = selector.select(*channel_actions)
            if action == "stop":
                return
            else:
                action._recv(channel)

    def poll_c_outports(self):
        while not self.grpc_stopping_event.is_set():
            # Need to poll both GRPC and CSP ports for messages
            # Service read write requests here
            for channel, cport in self.c_outports_to_poll.items():
                if channel.probe() > 0:
                    cport._send(channel)

    def _create_channel(self,
                        channel_name: str,
                        message_size: int,
                        number_elements: int,
                        slack: int,
                        host_idx: int = 0):
        return self.board.hosts[host_idx].createChannel(
            name=channel_name,
            messageSize=message_size,
            numElements=number_elements,
            slack=slack
        )

    def create_channel(self,
                       input_channel: bool,
                       port_idx: int,
                       c_port: AbstractCPort,
                       snip: EmbeddedSnip,
                       c_builder_idx: int) -> ty.List[Channel]:
        channels: ty.List[Channel] = []
        MESSAGE_SIZE_IN_C = 128 * 4
        DEFAULT_CHANNEL_SLACK = 16
        channel_slack = self._compile_config.get("channel_slack",
                                                 DEFAULT_CHANNEL_SLACK)
        if input_channel:
            for csp_port in c_port.csp_ports:
                channel_name = generate_channel_name("in_grpc_",
                                                     port_idx,
                                                     csp_port,
                                                     c_builder_idx)
                channels.append(self._create_channel(
                    channel_name=channel_name,
                    message_size=MESSAGE_SIZE_IN_C,
                    number_elements=csp_port.size,
                    slack=channel_slack
                ))
        else:
            for csp_port in c_port.csp_ports:
                channel_name = generate_channel_name("out_grpc_",
                                                     port_idx,
                                                     csp_port,
                                                     c_builder_idx)
                channels.append(self._create_channel(
                    channel_name=channel_name,
                    message_size=MESSAGE_SIZE_IN_C,
                    number_elements=csp_port.size,
                    slack=channel_slack
                ))

        if input_channel:
            for channel in channels:
                channel.connect(None, snip)
                self.c_inports_to_poll[c_port] = channel
        else:
            for channel in channels:
                channel.connect(snip, None)
                self.c_outports_to_poll[channel] = c_port

        return channels
