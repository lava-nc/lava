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


def realsense_msg_process(res):
    stamp = int.from_bytes(bytearray(np.flipud(res[0:8]).tolist()),
                           byteorder='big', signed=False)
    channel = int.from_bytes(bytearray(np.flipud(res[8:12]).tolist()),
                             byteorder='big', signed=False)
    width = int.from_bytes(bytearray(np.flipud(res[8:12:]).tolist()),
                           byteorder='big', signed=False)
    height = int.from_bytes(bytearray(np.flipud(res[12:16:]).tolist()),
                            byteorder='big', signed=False)
    img_data = res[16:]
    print("stamp nsec = ", stamp)
    print("channel = ", channel)
    print("width = ", width)
    print("height = ", height)
    print("img_data = ", img_data)
    # img = numpy2pil(img_data.reshape((height, width, channel)))
    # img.show()
    # img.close()
    # time.sleep(0.01)


def dvs_msg_process(res):
    stamp = int.from_bytes(bytearray(np.flipud(res[0:8]).tolist()),
                           byteorder='big', signed=False)
    width = int.from_bytes(bytearray(np.flipud(res[8:12:]).tolist()),
                           byteorder='big', signed=False)
    height = int.from_bytes(bytearray(np.flipud(res[12:16:]).tolist()),
                            byteorder='big', signed=False)
    print("stamp nsec = ", stamp)
    print("width = ", width)
    print("height = ", height)
    img_data = res[16:]
    img_data = img_data.reshape(height, width)
    print("img_data_shape = ", img_data.shape)
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            x = j
            y = i
            p = int(img_data[i][j])
            if p == 1:
                print("event_data [width = %d, height = %d, p = %d]"
                      % (x, y, p))
    dds_get_timestamp = time.time() * 1e9
    print("dds_get_timestamp =", dds_get_timestamp)
    print("dvs_ros_latency = %ld nsec", dds_get_timestamp - stamp)


def test_ddschannel():
    name = 'rt/prophesee/PropheseeCamera_optical_frame/cd_events_buffer'

    dds_channel = GetDDSChannel(
        name,
        DDSTransportType.DDSUDPv4,
        DDSBackendType.FASTDDSBackend,
        ChannelQueueSize)

    recv_port = dds_channel.dst_port
    recv_port.start()
    for i in range(10):
        res = recv_port.recv()
        dvs_msg_process(res)
    recv_port.join()


if __name__ == "__main__":
    test_ddschannel()
