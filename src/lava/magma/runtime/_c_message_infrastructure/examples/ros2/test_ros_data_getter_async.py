# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import List
import numpy as np
from PIL import Image
import time

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyAsyncProcessModel
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.runtime.message_infrastructure import (
    ChannelQueueSize,
    GetDDSChannel,
    DDSTransportType,
    DDSBackendType
)


def numpy2pil(np_array: np.ndarray) -> Image:
    img = Image.fromarray(np_array, 'RGB')
    return img


class RosGetterRunConfig(RunConfig):
    def __init__(self, select_tag: str = 'rs_frame'):
        super(RosGetterRunConfig, self).__init__(custom_sync_domains=None)
        self.select_tag = select_tag

    def select(
        self, _, proc_models: List[PyAsyncProcessModel]
    ) -> PyAsyncProcessModel:
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError('No legal ProcessModel found.')


class RosFrameGetterProcess(AbstractProcess):
    """Realsense Frame getter object."""
    def __init__(self, topic: str, num_step: int) -> None:
        super().__init__(topic=topic, num_step=num_step)


@implements(proc=RosFrameGetterProcess, protocol=AsyncProtocol)
@tag("rs_frame")
@requires(CPU)
class RosGetterProcModel(PyAsyncProcessModel):
    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.sample_time = 0
        self.topic = self.proc_params['topic']
        self.num_step = self.proc_params['num_step']
        self.dds_channel = GetDDSChannel(
            self.topic,
            DDSTransportType.DDSUDPv4,
            DDSBackendType.FASTDDSBackend,
            ChannelQueueSize
        )
        self.dst_port = self.dds_channel.dst_port
        self.dst_port.start()
        self.current_step = 0

    def run_async(self) -> None:
        while self.num_step > self.current_step:
            res = self.dst_port.recv()
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

            # Stops async process when desired number of steps is reached
            self.current_step += 1

            # Stops async process if stop command is sent by runtime
            if self.check_for_stop_cmd():
                self.dst_port.join()


def test_dds_from_ros_for_realsense():
    topic = 'rt/camera/color/image_raw_dds'
    num_steps = 10
    proc = RosFrameGetterProcess(topic=topic, num_step=num_steps)
    run_condition = RunSteps(num_steps=num_steps)
    run_config = RosGetterRunConfig(select_tag='rs_frame')
    proc.run(condition=run_condition, run_cfg=run_config)
    proc.stop()


if __name__ == "__main__":
    test_dds_from_ros_for_realsense()
