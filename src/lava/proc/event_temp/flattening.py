# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import math
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


class Flattening(AbstractProcess):
    def __init__(self,
                 *,
                 shape_in: ty.Union[ty.Tuple[int, int], ty.Tuple[int, int, int]],
                 **kwargs) -> None:
        super().__init__(shape_in=shape_in,
                         **kwargs)

        self._validate_shape_in(shape_in)

        shape_out = (math.prod(shape_in),)

        self.in_port = InPort(shape_in)
        self.out_port = OutPort(shape_out)

    @staticmethod
    def _validate_shape_in(shape_in: ty.Union[ty.Tuple[int, int], ty.Tuple[int, int, int]]) -> None:
        if not (len(shape_in) == 2 or len(shape_in) == 3):
            raise ValueError(f"shape_in should be 2 or 3 dimensional. "
                             f"{shape_in} was given.")

        if len(shape_in) == 3:
            if shape_in[2] != 2:
                raise ValueError(f"Third dimension of shape_in should be "
                                 f"equal to 2."
                                 f"{shape_in} was given.")

        if shape_in[0] <= 0 or shape_in[1] <= 0:
            raise ValueError(f"Width and height of shape_in should be positive."
                             f"{shape_in} was given.")


@implements(proc=Flattening, protocol=LoihiProtocol)
@requires(CPU)
class FlatteningPM(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self._shape_in = proc_params["shape_in"]

    def run_spk(self) -> None:
        data = self.in_port.recv()
        self.out_port.send(data.flatten())
