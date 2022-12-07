# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from numpy.lib.stride_tricks import as_strided
import typing as ty

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.max_pooling.process import MaxPooling


@implements(proc=MaxPooling, protocol=LoihiProtocol)
@requires(CPU)
class PyMaxPoolingPM(PyLoihiProcessModel):
    """PyLoihiProcessModel implementing the MaxPooling Process.

    Applies the max-pooling operation on incoming data.
    """
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    kernel_size: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=8)
    stride: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=8)
    padding: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=8)

    def run_spk(self) -> None:
        data = self.in_port.recv()

        max_pooled_data = self._max_pooling(data)

        self.out_port.send(max_pooled_data)

    def _max_pooling(self, data: np.ndarray) -> np.ndarray:
        """Applies the max-pooling operation on data with shape (W, H, C).

        Parameters
        ----------
        data : np.ndarray
            Incoming data.

        Returns
        ----------
        result : np.ndarray
            3D result after max-pooling.
        """
        result = np.zeros(self.out_port.shape)

        for channel in range(self.out_port.shape[-1]):
            result[:, :, channel] = \
                self._max_pooling_2d(data[:, :, channel],
                                     output_shape=self.out_port.shape[:-1],
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding)

        return result

    @staticmethod
    def _max_pooling_2d(data: np.ndarray,
                        output_shape: ty.Tuple[int, int],
                        kernel_size: np.ndarray,
                        stride: np.ndarray,
                        padding: np.ndarray) -> np.ndarray:
        """Applies the max-pooling operation on data with shape (W, H).

        Parameters
        ----------
        data : np.ndarray
            Data with shape (W, H).
        output_shape : tuple(int, int)
            Output shape.
        kernel_size : np.ndarray
            Max-pooling kernel size.
        stride : np.ndarray
            Max-pooling stride.
        padding : np.ndarray
            Padding to apply.

        Returns
        ----------
        result : np.ndarray
            2D result after max-pooling.
        """
        padded_data = np.pad(data,
                             (padding[0], padding[1]),
                             mode='constant').copy()

        shape_w = (output_shape[0],
                   output_shape[1],
                   kernel_size[0],
                   kernel_size[1])
        strides_w = (stride[0] * padded_data.strides[0],
                     stride[1] * padded_data.strides[1],
                     padded_data.strides[0],
                     padded_data.strides[1])

        pooled_data = as_strided(padded_data, shape_w, strides_w)
        max_pooled_data = pooled_data.max(axis=(2, 3))

        return max_pooled_data
