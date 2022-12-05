# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal
import typing as ty

from lava.proc.event_data.event_pre_processor.utils import \
    DownSamplingMethodDense

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class DownSamplingDense(AbstractProcess):
    def __init__(self,
                 shape_in: ty.Tuple[int],
                 down_sampling_method: DownSamplingMethodDense,
                 down_sampling_factor: int,
                 **kwargs) -> None:
        super().__init__(shape_in=shape_in,
                         down_sampling_method=down_sampling_method,
                         down_sampling_factor=down_sampling_factor,
                         **kwargs)

        self._validate_shape_in(shape_in)
        self._validate_down_sampling_method(down_sampling_method)
        self._validate_down_sampling_factor(down_sampling_factor)
        # test invalid shape in (negative/decimal values, 1d, 4+d, 3rd dim not 2)
        # test for invalid down sampling factor (negative values)
        # test for invalid type given to down sampling method

        shape_out = (shape_in[0] // down_sampling_factor,
                     shape_in[1] // down_sampling_factor)
        self.in_port = InPort(shape=shape_in)
        self.out_port = OutPort(shape=shape_out)

    @staticmethod
    def _validate_shape_in(shape_in):
        if not (len(shape_in) == 2 or len(shape_in) == 3):
            raise ValueError(f"shape_in should be 2 or 3 dimensional. "
                             f"{shape_in} given.")

        if not isinstance(shape_in[0], int) or not isinstance(shape_in[1], int):
            raise ValueError(f"Width and height of shape_in should be integers."
                             f"{shape_in} given.")
        if len(shape_in) == 3:
            if shape_in[2] != 2:
                raise ValueError(f"Third dimension of shape_in should be "
                                 f"equal to 2. "
                                 f"{shape_in} given.")

        if shape_in[0] <= 0 or shape_in[1] <= 0:
            raise ValueError(f"Width and height of shape_in should be positive."
                             f"{shape_in} given.")

        return shape_in

    @staticmethod
    def _validate_down_sampling_method(down_sampling_method):
        if not isinstance(down_sampling_method, DownSamplingMethodDense):
            raise (TypeError(
                f"Down sampling methods for dense to dense down-sampling need to be "
                f"selected using the DownSamplingMethodDense Enum."))
            # TODO: mention that it's an enum in error message?

    @staticmethod
    def _validate_down_sampling_factor(down_sampling_factor):
        # TODO: should the down sampling factor be a float or an int?
        if not isinstance(down_sampling_factor, int):
            raise (ValueError(f"Down sampling factor should be an integer."
                              f"{down_sampling_factor} given."))

        if down_sampling_factor <= 0:
            raise ValueError(f"Down sampling factor should be positive."
                             f"{down_sampling_factor} given.")


@implements(proc=DownSamplingDense, protocol=LoihiProtocol)
@requires(CPU)
class DownSamplingDensePM(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self._shape_in = proc_params["shape_in"]
        self._down_sampling_method = proc_params["down_sampling_method"]
        self._down_sampling_factor = proc_params["down_sampling_factor"]

        self._shape_out = (self._shape_in[0] // self._down_sampling_factor,
                           self._shape_in[1] // self._down_sampling_factor)

    def run_spk(self) -> None:
        data = self.in_port.recv()

        down_sampled_data = self._down_sample(data)

        self.out_port.send(down_sampled_data)

    def _down_sample(self, data: np.ndarray) -> np.ndarray:
        if self._down_sampling_method == DownSamplingMethodDense.SKIPPING:
            down_sampled_data = \
                self._down_sample_skipping(data,
                                           self._down_sampling_factor,
                                           self._shape_out[0],
                                           self._shape_out[1])

        elif self._down_sampling_method == DownSamplingMethodDense.MAX_POOLING:
            down_sampled_data = \
                self._down_sample_max_pooling(data,
                                              self._down_sampling_factor,
                                              self._shape_out[0],
                                              self._shape_out[1])

        elif self._down_sampling_method == DownSamplingMethodDense.CONVOLUTION:
            down_sampled_data = \
                self._down_sample_convolution(data,
                                              self._down_sampling_factor,
                                              self._shape_out[0],
                                              self._shape_out[1])

        else:
            # TODO : Remove since validation is taking care of this ?
            raise ValueError(f"Unknown down_sample_mode."
                             f"{self._down_sampling_method=} given.")

        return down_sampled_data

    @staticmethod
    def _down_sample_skipping(data: np.ndarray,
                              down_sampling_factor: int,
                              down_sampled_width: int,
                              down_sampled_height: int) -> np.ndarray:
        down_sampled_data = \
            data[::down_sampling_factor, ::down_sampling_factor]

        down_sampled_data = \
            down_sampled_data[:down_sampled_width, :down_sampled_height]

        return down_sampled_data

    @staticmethod
    def _down_sample_max_pooling(data: np.ndarray,
                                 down_sampling_factor: int,
                                 down_sampled_width: int,
                                 down_sampled_height: int) -> np.ndarray:
        output_shape = \
            ((data.shape[0] - down_sampling_factor) // down_sampling_factor + 1,
             (data.shape[1] - down_sampling_factor) // down_sampling_factor + 1)

        shape_w = (output_shape[0],
                   output_shape[1],
                   down_sampling_factor,
                   down_sampling_factor)
        strides_w = (down_sampling_factor * data.strides[0],
                     down_sampling_factor * data.strides[1],
                     data.strides[0],
                     data.strides[1])

        down_sampled_data = as_strided(data, shape_w, strides_w)
        down_sampled_data = down_sampled_data.max(axis=(2, 3))

        # TODO: Is this really needed ?
        down_sampled_data = \
            down_sampled_data[:down_sampled_width, :down_sampled_height]

        return down_sampled_data

    @staticmethod
    def _down_sample_convolution(data: np.ndarray,
                                 down_sampling_factor: int,
                                 down_sampled_width: int,
                                 down_sampled_height: int) -> np.ndarray:
        kernel = np.ones((down_sampling_factor, down_sampling_factor))
        data_convolved = signal.convolve2d(data, kernel)

        down_sampled_data = \
            data_convolved[::down_sampling_factor, ::down_sampling_factor]

        down_sampled_data = \
            down_sampled_data[:down_sampled_width, :down_sampled_height]

        return down_sampled_data
