# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2020 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
from dataclasses import dataclass
from enum import IntEnum, Enum
import typing as ty


class SpikeIOInterface(IntEnum):
    """Interface type for spike io communication"""
    ETHERNET = 0
    """10G Ethernet"""
    PIO = 1
    """FPGA/PIO"""


class SpikeIOPort(Enum):
    """Spike IO Port Types"""
    PIO_NORTH = 'n'
    PIO_SOUTH = 's'
    PIO_EAST = 'e'
    PIO_WEST = 'w'
    PIO_UP = 'u'
    PIO_DOWN = 'd'
    ETHERNET = 'p'


class SpikeIOMode(Enum):
    """Modes of Spike IO"""
    FREE_RUNNING = 0
    TIME_COMPARE = 1


@dataclass
class ConnectionConfig:
    """Configuration class for a Connection Instance"""
    interface: SpikeIOInterface = SpikeIOInterface.ETHERNET
    num_probes: int = 4
    ethernet_packet_len: int = 256
    ethernet_interface: str = "enp2s0"
    max_messages: int = 1024
    max_message_size: int = 4096
    spike_io_port: SpikeIOPort = SpikeIOPort.ETHERNET
    spike_io_mode: SpikeIOMode = SpikeIOMode.TIME_COMPARE
    num_time_buckets: int = 1 << 16
    ethernet_mac_address: str = "0x90e2ba01214c"
    loihi_mac_address: str = "0x0015edbeefed"
    ethernet_chip_id: ty.Optional[ty.Tuple[int, int, int]] = None
    ethernet_chip_idx: ty.Optional[int] = None
