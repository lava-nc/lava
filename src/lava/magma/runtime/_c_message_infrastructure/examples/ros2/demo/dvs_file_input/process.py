# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from PIL import Image


from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.magma.runtime.message_infrastructure import (
    ChannelQueueSize,
    GetDDSChannel,
    DDSTransportType,
    DDSBackendType
)

from scipy.ndimage import gaussian_filter

def numpy2pil(np_array: np.ndarray, fm = 'RGB') -> Image:
    img = Image.fromarray(np_array, fm)
    return img

"""# Input Encoder ###############################################################
class PilotNetEncoder(AbstractProcess):
    def __init__(self,
                 shape: Tuple[int, ...],
                 vth: int,
                 compression: Compression = Compression.DELTA_SPARSE_8) -> None:
        super().__init__(shape=shape,
                         vth=vth,
                         compression=compression)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=PilotNetEncoder, protocol=LoihiProtocol)
@requires(CPU)
class PilotNetPyEncoderModel(AbstractSubProcessModel):
    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int64)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int64)
        shape = proc.proc_params.get('shape')
        vth = proc.proc_params.get('vth')
        self.encoder = io.encoder.DeltaEncoder(shape=shape,
                                               vth=vth,
                                               spike_exp=6)
        proc.inp.connect(self.encoder.a_in)
        self.encoder.s_out.connect(proc.out)"""


class DVSFileInput(AbstractProcess):
    """outputting a frame from ROS camera fetched from DDSChannel, after converting it
        to events"""
    def __init__(self,
                 true_height: int,
                 true_width: int,
                 down_sample_factor: int = 1,
                 num_steps=1, 
                 diff_thresh = 10) -> None:
        super().__init__(true_height=true_height,
                         true_width=true_width,
                         down_sample_factor=down_sample_factor,
                         num_steps=num_steps,
                         diff_thresh=diff_thresh)

        down_sampled_height = true_height // down_sample_factor
        down_sampled_width = true_width // down_sample_factor

        out_shape = (down_sampled_width, down_sampled_height)
        self.event_frame_out = OutPort(shape=out_shape)


@implements(proc=DVSFileInput, protocol=LoihiProtocol)
@requires(CPU)
class PyDVSFileInputPM(PyLoihiProcessModel):
    event_frame_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._true_height = proc_params["true_height"]
        self._true_width = proc_params["true_width"]
        self._true_shape = (self._true_width, self._true_height)

        self._down_sample_factor = proc_params["down_sample_factor"]
        self._down_sampled_height = \
            self._true_height // self._down_sample_factor
        self._down_sampled_width = \
            self._true_width // self._down_sample_factor
        self._down_sampled_shape = (self._down_sampled_width,
                                    self._down_sampled_height)
        self._frame_shape = self._down_sampled_shape[::-1]
                                    
        self._num_steps = proc_params["num_steps"]
        self._cur_steps = 0

        # DDSChannel relavent
        name = 'rt/camera/color/image_raw_dds'

        self.dds_channel = GetDDSChannel(
            name,
            DDSTransportType.DDSUDPv4,
            DDSBackendType.FASTDDSBackend,
            ChannelQueueSize)

        self.recv_port = self.dds_channel.dst_port
        self.recv_port.start()

        # Frame comparison relv
        self.diff_thresh = proc_params["diff_thresh"]
        self.prev_frame = np.zeros((*self._frame_shape, 3))

        self.saved = 0


    def diff(self, frame1, frame2):
        real_diffs = frame2 - frame1
        # return (np.absolute(real_diffs[:,:,:]) > self.diff_thresh).sum(axis=2) > 0  # comparing the absolute value
        return ((frame2 * (1 / 3)).sum(axis = 2)) / 255   # grayscale
        # return np.linalg.norm(real_diffs, axis=2) > self.diff_thresh # distance

    def run_spk(self):
        self._cur_steps += 1
        
        res = self.recv_port.recv()
        """stamp = int.from_bytes(bytearray(res[0:8].tolist()),
                               byteorder='big', signed=False)
        channel = int.from_bytes(bytearray(res[8:12].tolist()),
                                 byteorder='big', signed=False)"""
        width = int.from_bytes(bytearray(res[12:16].tolist()),
                               byteorder='big', signed=False)
        height = int.from_bytes(bytearray(res[16:20].tolist()),
                                byteorder='big', signed=False)
        img_data = res[20:]
        img_data = img_data.reshape((height, width, 3))

        # apply gaussian blur
        event_frame_small = \
                self._convolution(img_data)
        diff = self.diff(self.prev_frame, event_frame_small)
        """if self.saved == 2:
            # img = numpy2pil(np.uint8(diff * 255), fm='L')
            img = numpy2pil(img_data)
            img.save("/home/zuom/img.jpg")
            print(self.prev_frame)
            print("========================")
            print(event_frame_small)
            print("=========================")
            print(event_frame_small - self.prev_frame)
            print("=========================")
            print(np.linalg.norm(event_frame_small - self.prev_frame, axis=2))
            print("=========================")
            print("threashold ", self.diff_thresh)
        self.saved += 1"""

        self.prev_frame = event_frame_small
            
        diff = np.transpose(diff, axes=(1, 0)) #* later codes assume (width, height) shaped image
        self.event_frame_out.send(diff)

    def post_guard(self) -> bool:
        return self._cur_steps == self._num_steps

    def run_post_mgmt(self) -> None:
        print("managed")
        self.recv_port.join()

    # gaussian blur
    def _convolution(self, matrix: np.ndarray, kernel_size: int = 16):
        event_frame_convolved = gaussian_filter(matrix, 8, radius=kernel_size)
        
        event_frame_small = \
            event_frame_convolved[::self._down_sample_factor,
                                  ::self._down_sample_factor]

        return event_frame_small