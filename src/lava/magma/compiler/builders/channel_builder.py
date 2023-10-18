# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/

import typing as ty
from dataclasses import dataclass
from multiprocessing import Queue

from lava.magma.compiler.builders.interfaces import \
    AbstractChannelBuilder, \
    AbstractProcessModel
from lava.magma.compiler.builders. \
    runtimeservice_builder import RuntimeServiceBuilder
from lava.magma.compiler.channels.interfaces import (
    Channel,
    ChannelType,
)
from lava.magma.compiler.utils import PortInitializer
from lava.magma.runtime.message_infrastructure \
    .message_infrastructure_interface import (MessageInfrastructureInterface)
from lava.magma.compiler.channels.watchdog import WatchdogManager, Watchdog

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.runtime.runtime import Runtime

Queues = ty.Tuple[Queue, Queue, Queue, Queue]
Watchdogs = ty.Tuple[Watchdog, Watchdog, Watchdog, Watchdog]
PortInitializers = ty.Tuple[PortInitializer, PortInitializer]


class WatchdogEnabledMixin:
    SEND = "send"
    RECV = "recv"
    JOIN = "join"

    @staticmethod
    def watch(watchdog_manager: WatchdogManager,
              queue: Queue,
              process: "AbstractProcess",
              other_process: "AbstractProcess",
              pi: PortInitializer,
              other_pi: PortInitializer,
              method_type: str) -> Watchdog:
        process_cls: str = process.__class__.__name__
        port_name: str = pi.name
        name: str = f"{process_cls}.{port_name}"
        if other_process and other_pi:
            other_process_cls: str = other_process.__class__.__name__
            other_port_name: str = other_pi.name
            other_name: str = f"{other_process_cls}.{other_port_name}"
            if method_type is WatchdogEnabledMixin.SEND:
                name += f"->{other_name}"
            elif method_type is WatchdogEnabledMixin.RECV:
                name = f"{other_name}->{name}"
        w: Watchdog = watchdog_manager.create_watchdog(queue=queue,
                                                       channel_name=name,
                                                       method_type=method_type)
        return w

    def create_watchdogs(self,
                         watchdog_manager: WatchdogManager,
                         queues: Queues,
                         port_initializers: PortInitializers) -> Watchdogs:
        src_send_watchdog: Watchdog = self.watch(watchdog_manager,
                                                 queues[0],
                                                 self.src_process,
                                                 self.dst_process,
                                                 port_initializers[0],
                                                 port_initializers[1],
                                                 WatchdogEnabledMixin.SEND)
        src_join_watchdog: Watchdog = self.watch(watchdog_manager,
                                                 queues[1],
                                                 self.src_process,
                                                 self.dst_process,
                                                 port_initializers[0],
                                                 port_initializers[1],
                                                 WatchdogEnabledMixin.JOIN)
        dst_recv_watchdog: Watchdog = self.watch(watchdog_manager,
                                                 queues[2],
                                                 self.dst_process,
                                                 self.src_process,
                                                 port_initializers[1],
                                                 port_initializers[0],
                                                 WatchdogEnabledMixin.RECV)
        dst_join_watchdog: Watchdog = self.watch(watchdog_manager,
                                                 queues[3],
                                                 self.dst_process,
                                                 self.src_process,
                                                 port_initializers[1],
                                                 port_initializers[0],
                                                 WatchdogEnabledMixin.JOIN)
        return (src_send_watchdog, src_join_watchdog,
                dst_recv_watchdog, dst_join_watchdog)


@dataclass
class ChannelBuilderMp(AbstractChannelBuilder, WatchdogEnabledMixin):
    """A ChannelBuilder assuming Python multi-processing is used as messaging
    and multi processing backbone.
    """

    channel_type: ChannelType
    src_process: "AbstractProcess"
    dst_process: "AbstractProcess"
    src_port_initializer: PortInitializer
    dst_port_initializer: PortInitializer

    def build(
            self, messaging_infrastructure: MessageInfrastructureInterface,
            watchdog_manager: WatchdogManager
    ) -> Channel:
        """Given the message passing framework builds a channel

        Parameters
        ----------
        messaging_infrastructure : MessageInfrastructureInterface
        watchdog_manager: WatchdogManager

        Returns
        -------
        Channel
            Channel

        Raises
        ------
        Exception
            Can't build channel of type specified
        """
        channel_class = messaging_infrastructure.channel_class(
            channel_type=self.channel_type
        )

        # Watchdogs
        sq = watchdog_manager.sq
        queues = (sq, sq, sq, sq)
        port_initializers = (self.src_port_initializer,
                             self.dst_port_initializer)
        (src_send_watchdog, src_join_watchdog,
         dst_recv_watchdog, dst_join_watchdog) = \
            self.create_watchdogs(watchdog_manager, queues, port_initializers)

        return channel_class(
            messaging_infrastructure,
            self.src_port_initializer.name,
            self.dst_port_initializer.name,
            self.src_port_initializer.shape,
            self.src_port_initializer.d_type,
            self.src_port_initializer.size,
            src_send_watchdog, src_join_watchdog,
            dst_recv_watchdog, dst_join_watchdog
        )


@dataclass
class ServiceChannelBuilderMp(AbstractChannelBuilder, WatchdogEnabledMixin):
    """A RuntimeServiceChannelBuilder assuming Python multi-processing is used
    as messaging and multi processing backbone.
    """

    channel_type: ChannelType
    src_process: ty.Union[RuntimeServiceBuilder,
                          ty.Type["AbstractProcessModel"]]
    dst_process: ty.Union[RuntimeServiceBuilder,
                          ty.Type["AbstractProcessModel"]]
    port_initializer: PortInitializer

    def build(
            self, messaging_infrastructure: MessageInfrastructureInterface,
            watchdog_manager: WatchdogManager
    ) -> Channel:
        """Given the message passing framework builds a channel

        Parameters
        ----------
        messaging_infrastructure : MessageInfrastructureInterface
        watchdog_manager: WatchdogManager

        Returns
        -------
        Channel
            PyPyChannel

        Raises
        ------
        Exception
            Can't build channel of type specified
        """
        channel_class = messaging_infrastructure.channel_class(
            channel_type=self.channel_type
        )

        # Watchdogs
        lq, sq = watchdog_manager.lq, watchdog_manager.sq
        queues = (sq, sq, lq, sq)
        port_initializers = (self.port_initializer,
                             self.port_initializer)
        (src_send_watchdog, src_join_watchdog,
         dst_recv_watchdog, dst_join_watchdog) = \
            self.create_watchdogs(watchdog_manager, queues, port_initializers)

        channel_name: str = self.port_initializer.name
        return channel_class(
            messaging_infrastructure,
            channel_name + "_src",
            channel_name + "_dst",
            self.port_initializer.shape,
            self.port_initializer.d_type,
            self.port_initializer.size,
            src_send_watchdog, src_join_watchdog,
            dst_recv_watchdog, dst_join_watchdog
        )


@dataclass
class RuntimeChannelBuilderMp(AbstractChannelBuilder, WatchdogEnabledMixin):
    """A RuntimeChannelBuilder assuming Python multi-processing is
    used as messaging and multi processing backbone.
    """

    channel_type: ChannelType
    src_process: ty.Union[RuntimeServiceBuilder, ty.Type["Runtime"]]
    dst_process: ty.Union[RuntimeServiceBuilder, ty.Type["Runtime"]]
    port_initializer: PortInitializer

    def build(
            self, messaging_infrastructure: MessageInfrastructureInterface,
            watchdog_manager: WatchdogManager
    ) -> Channel:
        """Given the message passing framework builds a channel

        Parameters
        ----------
        messaging_infrastructure : MessageInfrastructureInterface
        watchdog_manager: WatchdogManager

        Returns
        -------
        Channel
            PyPyChannel

        Raises
        ------
        Exception
            Can't build channel of type specified
        """
        channel_class = messaging_infrastructure.channel_class(
            channel_type=self.channel_type
        )

        # Watchdogs
        lq, sq = watchdog_manager.lq, watchdog_manager.sq
        queues = (sq, sq, lq, sq)
        port_initializers = (self.port_initializer,
                             self.port_initializer)
        (src_send_watchdog, src_join_watchdog,
         dst_recv_watchdog, dst_join_watchdog) = \
            self.create_watchdogs(watchdog_manager, queues, port_initializers)

        channel_name: str = self.port_initializer.name
        return channel_class(
            messaging_infrastructure,
            channel_name + "_src",
            channel_name + "_dst",
            self.port_initializer.shape,
            self.port_initializer.d_type,
            self.port_initializer.size,
            src_send_watchdog, src_join_watchdog,
            dst_recv_watchdog, dst_join_watchdog
        )


@dataclass
class ChannelBuilderNx(AbstractChannelBuilder):
    """A ChannelBuilder for CNc and NcC Channels with NxBoard as
    the messaging
    infrastructure.
    """

    channel_type: ChannelType
    src_process: "AbstractProcess"
    dst_process: "AbstractProcess"
    src_port_initializer: PortInitializer
    dst_port_initializer: PortInitializer

    def build(
            self, messaging_infrastructure: MessageInfrastructureInterface
    ) -> Channel:
        """Given the message passing framework builds a channel

        Parameters
        ----------
        messaging_infrastructure : MessageInfrastructureInterface

        Returns
        -------
        Channel
            Channel

        Raises
        ------
        Exception
            Can't build channel of type specified
        """


class ChannelBuilderPyNc(ChannelBuilderNx):
    """A ChannelBuilder for PyNc and NcPy Channels with NxBoard as the messaging
    infrastructure.
    """
    def build(
            self, messaging_infrastructure: MessageInfrastructureInterface
    ) -> Channel:
        channel_class = messaging_infrastructure.channel_class(
            channel_type=self.channel_type
        )

        return channel_class(
            self.src_port_initializer.name,
            self.dst_port_initializer.name,
            self.src_port_initializer.shape,
            self.src_port_initializer.d_type,
            self.src_port_initializer.size,
            self.src_port_initializer,
            self.dst_port_initializer
        )
