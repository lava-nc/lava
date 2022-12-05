# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class SparseToDense(AbstractProcess):
    def __init__(self,
                 *,
                 shape_in: ty.Tuple[int],
                 shape_out: ty.Union[ty.Tuple[int, int], ty.Tuple[int, int, int]],
                 **kwargs) -> None:
        super().__init__(shape_in=shape_in,
                         shape_out=shape_out,
                         **kwargs)

        self._validate_shape_in(shape_in)
        self._validate_shape_out(shape_out)

        self.in_port = InPort(shape=shape_in)
        self.out_port = OutPort(shape=shape_out)

    @staticmethod
    def _validate_shape_in(shape_in: ty.Tuple[int]) -> None:
        if len(shape_in) != 1:
            raise ValueError(f"Shape of the InPort should be (n,). "
                             f"{shape_in} was given.")

        if shape_in[0] <= 0:
            raise ValueError(f"Width of shape_in should be positive. {shape_in} given.")

    @staticmethod
    def _validate_shape_out(shape_out: ty.Union[ty.Tuple[int, int], ty.Tuple[int, int, int]]) -> None:
        if not (len(shape_out) == 2 or len(shape_out) == 3):
            raise ValueError(f"shape_out should be 2 or 3 dimensional. {shape_out} given.")

        if len(shape_out) == 3:
            if shape_out[2] != 2:
                raise ValueError(f"Depth of the shape_out argument should be an integer and equal to 2. "
                                 f"{shape_out} given.")

        if shape_out[0] <= 0 or shape_out[1] <= 0:
            raise ValueError(f"Width and height of the shape_out argument should be positive. "
                             f"{shape_out} given.")


@implements(proc=SparseToDense, protocol=LoihiProtocol)
@requires(CPU)
class SparseToDensePM(PyLoihiProcessModel):
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
        # TODO : Should we add an else here ?
        # TODO : We will never reach it if correctly validated

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
