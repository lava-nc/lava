# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.event_data.to_frame.process import ToFrame


@implements(proc=ToFrame, protocol=LoihiProtocol)
@requires(CPU)
class ToFramePM(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self._shape_out = proc_params["shape_out"]

    def run_spk(self) -> None:
        data, indices = self.in_port.recv()

        dense_data = self._transform(data, indices)

        self.out_port.send(dense_data)

    def _transform(self, data: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if len(self._shape_out) == 2:
            return self._transform_2d(data, indices)
        elif len(self._shape_out) == 3:
            return self._transform_3d(data, indices)

    def _transform_2d(self,
                      data: np.ndarray,
                      indices: np.ndarray) -> np.ndarray:
        dense_data = np.zeros(self._shape_out)

        xs, ys = np.unravel_index(indices, self._shape_out)

        dense_data[xs[data == 0], ys[data == 0]] = 1
        dense_data[xs[data == 1], ys[data == 1]] = 1

        return dense_data

    def _transform_3d(self,
                      data: np.ndarray,
                      indices: np.ndarray) -> np.ndarray:
        dense_data = np.zeros(self._shape_out)

        xs, ys = np.unravel_index(indices, self._shape_out[:-1])

        dense_data[xs[data == 0], ys[data == 0], 0] = 1
        dense_data[xs[data == 1], ys[data == 1], 1] = 1

        return dense_data
