# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from functools import partial
import time

from lava.magma.runtime.message_infrastructure.multiprocessing \
    import MultiProcessing

from lava.magma.runtime.message_infrastructure import (
    ChannelBackend,
    Channel,
    SendPort,
    RecvPort,
    ChannelQueueSize,
    GetDDSChannel,
    DDSTransportType,
    DDSBackendType
)


def prepare_data():
    return np.random.random_sample((2, 4))


def test_ddschannel():
    name = 'rt/dds_topic'

    dds_channel = GetDDSChannel(
        name,
        DDSTransportType.DDSUDPv4,
        DDSBackendType.CycloneDDSBackend,
        ChannelQueueSize)

    send_port = dds_channel.src_port
    send_port.start()
    for i in range(100):
        send_port.send(prepare_data())
        time.sleep(0.1)
    send_port.join()


if __name__ == "__main__":
    test_ddschannel()
