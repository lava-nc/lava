# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from functools import partial
import time

from message_infrastructure.multiprocessing import MultiProcessing

from message_infrastructure import (
    ChannelBackend,
    Channel,
    SendPort,
    RecvPort,
    SupportDDSChannel,
    ChannelQueueSize,
    GetDDSChannel,
    DDSTransportType,
    DDSBackendType
)

def test_ddschannel(self):
    nbytes = np.prod(const_data.shape) * const_data.dtype.itemsize
    name = 'test_dds_to_ros'

    dds_channel = GetDDSChannel(
        name,
        # DDSTransportType.DDSSHM,
        DDSTransportType.DDSUDPv4,
        DDSBackendType.FASTDDSBackend,
        ChannelQueueSize)

    recv_port = dds_channel.dst_port
    recv_port.start()
    for i in range(100):
        res = recv_port.recv()
        print(res)
    recv_port.join()