# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import Iterable, Tuple, Union, List
import numpy as np
from PIL import Image

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import HostCPU, CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
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
        self, _, proc_models: List[PyLoihiProcessModel]
    ) -> PyLoihiProcessModel:
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError('No legal ProcessModel found.')


class RosFrameGetterProcess(AbstractProcess):
    """Realsense Frame getter object."""
    def __init__(
        self,
        topic: str,
        num_step: int,
    ) -> None:
        super().__init__(topic=topic, num_step=num_step)


@implements(proc=RosFrameGetterProcess, protocol=LoihiProtocol)
@tag("rs_frame")
@requires(CPU)
class RosGetterProcModel(PyLoihiProcessModel):
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

    def run_spk(self) -> None:
        res = self.dst_port.recv()
        stamp = int.from_bytes(bytearray(res[0:8].tolist()),
                               byteorder='big', signed=False)
        channel = int.from_bytes(bytearray(res[8:12].tolist()),
                                 byteorder='big', signed=False)
        width = int.from_bytes(bytearray(res[12:16].tolist()),
                               byteorder='big', signed=False)
        height = int.from_bytes(bytearray(res[16:20].tolist()),
                                byteorder='big', signed=False)
        img_data = res[20:].sum()
        print("stamp nsec = ", stamp)
        print("channel = ", channel)
        print("width = ", width)
        print("height = ", height)
        print("img_data = ", img_data)
        img = numpy2pil(img_data.reshape((height, width, channel)))
        img.show()
        img.close()

    def post_guard(self) -> bool:
        return self.time_step == self.num_step

    def run_post_mgmt(self) -> None:
        self.dst_port.join()


def test_dds_from_ros_for_realsense():
    topic = 'rt/camera/depth/image_rect_raw_dds'
    num_steps = 10
    proc = RosFrameGetterProcess(topic=topic, num_step=num_steps)
    run_condition = RunSteps(num_steps=num_steps)
    run_config = RosGetterRunConfig(select_tag='rs_frame')
    proc.run(condition=run_condition, run_cfg=run_config)
    proc.stop()


if __name__ == "__main__":
    test_dds_from_ros_for_realsense()
