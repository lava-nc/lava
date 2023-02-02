# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest
from functools import partial
import time
from PIL import Image
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

def numpy2pil(np_array: np.ndarray) -> Image:
    img = Image.fromarray(np_array, 'RGB')
    return img


def test_ddschannel():
    name = 'rt/camera/color/image_raw_dds'

    dds_channel = GetDDSChannel(
        name,
        DDSTransportType.DDSUDPv4,
        DDSBackendType.FASTDDSBackend,
        ChannelQueueSize)

    recv_port = dds_channel.dst_port
    recv_port.start()
    for i in range(100):
        res = recv_port.recv()
        print(res.size)
        res = res.reshape((480, 640, 3))
        img = numpy2pil(res)
        img.show()
        img.close()
        time.sleep(1)
    recv_port.join()


if __name__ == "__main__":
    test_ddschannel()
