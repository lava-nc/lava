# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import time
from PIL import Image

from lava.magma.runtime.message_infrastructure import (
    ChannelQueueSize,
    GetDDSChannel,
    DDSTransportType,
    DDSBackendType
)


def numpy2pil(np_array: np.ndarray) -> Image:
    img = Image.fromarray(np_array, 'RGB')
    return img


def realsense_msg_process(res):
    stamp = int.from_bytes(bytearray(np.flipud(res[0:8]).tolist()),
                           byteorder='big', signed=False)
    channel = int.from_bytes(bytearray(np.flipud(res[8:12]).tolist()),
                             byteorder='big', signed=False)
    width = int.from_bytes(bytearray(np.flipud(res[12:16]).tolist()),
                           byteorder='big', signed=False)
    height = int.from_bytes(bytearray(np.flipud(res[16:20]).tolist()),
                            byteorder='big', signed=False)
    img_data = res[20:]
    print("stamp nsec = ", stamp)
    print("channel = ", channel)
    print("width = ", width)
    print("height = ", height)
    print("img_data = ", img_data)
    img = numpy2pil(img_data.reshape((height, width, channel)))
    img.show()
    img.close()
    time.sleep(0.1)


def test_ddschannel():
    name = 'rt/camera/color/image_raw_dds'

    dds_channel = GetDDSChannel(
        name,
        DDSTransportType.DDSUDPv4,
        DDSBackendType.FASTDDSBackend,
        ChannelQueueSize)

    recv_port = dds_channel.dst_port
    recv_port.start()
    for i in range(10):
        res = recv_port.recv()
        realsense_msg_process(res)
    recv_port.join()


if __name__ == "__main__":
    test_ddschannel()
