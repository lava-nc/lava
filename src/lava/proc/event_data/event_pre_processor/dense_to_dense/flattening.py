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


class Flattening(AbstractProcess):
    def __init__(self,
                 shape_in: tuple,
                 shape_out: tuple,
                 **kwargs) -> None:
        super().__init__(shape_in=shape_in,
                         shape_out=shape_out,
                         **kwargs)

        raise NotImplementedError()

        # TODO: Validation

        self.in_port = InPort()
        self.out_port = OutPort()


@implements(proc=Flattening, protocol=LoihiProtocol)
@requires(CPU)
class FlatteningPM(PyLoihiProcessModel):
    in_port: PyOutPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        raise NotImplementedError()

    def run_spk(self) -> None:
        raise NotImplementedError()

