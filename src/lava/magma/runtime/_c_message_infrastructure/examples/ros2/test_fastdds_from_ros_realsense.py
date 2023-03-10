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


def numpy2pil(np_array: np.ndarray, n_channels: int) -> Image:
    if n_channels == 3:
        img = Image.fromarray(np_array, mode='RGB')
    elif n_channels == 1:
        img = Image.fromarray(np_array, mode='L')
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
    print("img_data length = ", len(img_data))

    # Processing for Depth channel
    if channel == 1:
        # Reading depth image as unsigned 16-bit
        img_data = np.frombuffer(img_data, dtype=np.uint16)
        print("16-bit: ", img_data)
        # Downsample to unsigned 8-bit for PIL
        img_data_8bit = (img_data / img_data.max()) * 255
        img_numpy_array = img_data_8bit.reshape((height, width)).astype(np.uint8)
        print("8-bit: ", img_numpy_array)
    # Processing for Color channel
    elif channel == 3:
        img_numpy_array = img_data.reshape((height, width, channel))

    img = numpy2pil(img_numpy_array, channel)
    img.show()
    img.close()
    time.sleep(0.1)


def test_ddschannel():
    # Comment / uncomment the relevant topic to test
    # name = 'rt/camera/color/image_raw_dds'
    # name = 'rt/camera/aligned_depth_to_color/image_raw_dds'
    name = 'rt/camera/depth/image_rect_raw_dds'

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
