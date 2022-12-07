# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from numpy.lib.stride_tricks import as_strided

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.max_pooling.process import MaxPooling
from lava.proc.conv import utils


@implements(proc=MaxPooling, protocol=LoihiProtocol)
@requires(CPU)
class MaxPoolingPM(PyLoihiProcessModel):
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
        output_shape = self.out_port.shape

        padded_data = np.pad(data,
                             (utils.make_tuple(self.padding[0]),
                              utils.make_tuple(self.padding[1])),
                             mode='constant')

        shape_w = (output_shape[0],
                   output_shape[1],
                   self.kernel_size[0],
                   self.kernel_size[1])
        strides_w = (self.stride[0] * data.strides[0],
                     self.stride[1] * data.strides[1],
                     data.strides[0],
                     data.strides[1])

        pooled_data = as_strided(padded_data, shape_w, strides_w)
        max_pooled_data = pooled_data.max(axis=(2, 3))

        return max_pooled_data
