# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class BinaryToUnaryPolarity(AbstractProcess):
    def __init__(self,
                 shape: tuple,
                 **kwargs) -> None:
        super().__init__(shape=shape, **kwargs)

        self._validate_shape(shape)

        self.in_port = InPort(shape=shape)
        self.out_port = OutPort(shape=shape)

    @staticmethod
    def _validate_shape(shape):
        if not isinstance(shape[0], int):
            raise ValueError(f"Max number of events should be an integer."
                             f"{shape} given.")

        if shape[0] <= 0:
            raise ValueError(f"Max number of events should be positive. "
                             f"{shape} given.")

        if len(shape) != 1:
            raise ValueError(f"Shape of the OutPort should be 1D. "
                             f"{shape} given.")

        return shape


@implements(proc=BinaryToUnaryPolarity, protocol=LoihiProtocol)
@requires(CPU)
class BinaryToUnaryPolarityPM(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def run_spk(self) -> None:
        data, indices = self.in_port.recv()

        data = self._encode(data)

        self.out_port.send(data, indices)

    @staticmethod
    def _encode(data: np.ndarray) -> np.ndarray:
        data[data == 0] = 1

        return data

