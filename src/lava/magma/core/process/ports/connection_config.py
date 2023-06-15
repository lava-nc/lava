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
from enum import IntEnum


class SpikeIOInterface(IntEnum):
    """Interface type for spike io communication"""
    ETHERNET = 0
    """10G Ethernet"""
    PIO = 1
    """FPGA/PIO"""


@dataclass
class ConnectionConfig:
    interface: SpikeIOInterface = SpikeIOInterface.ETHERNET
    num_probes: int = 4
    ethernet_packet_len: int = 256
    ethernet_interface: str = "enp2s0"
    max_messages: int = 1024
    max_message_size: int = 4096
