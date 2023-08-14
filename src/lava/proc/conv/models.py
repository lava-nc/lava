# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.conv.process import Conv

from lava.proc.conv import utils


class AbstractPyConvModel(PyLoihiProcessModel):
    """Abstract template implementation of PyConvModel."""
    s_in = None
    a_out = None
    a_buf = None
    weight = None

    kernel_size: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=8)
    stride: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=8)
    padding: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=8)
    dilation: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=8)
    groups: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=8)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=5)
    weight_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=8)
    num_weight_bits: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=5)

    def run_spk(self) -> None:
        if self.a_buf is None:
            self.a_buf = np.zeros_like(self.a_out.shape)

        self.a_out.send(self.a_buf)

        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
        else:
            s_in = self.s_in.recv().astype(bool)

        a_out = utils.conv(
            s_in, self.weight,
            self.kernel_size, self.stride, self.padding, self.dilation,
            self.groups[0]
        )

        self.a_buf = self.clamp_precision(a_out)

    def clamp_precision(self, x: np.ndarray) -> np.ndarray:
        return x


@implements(proc=Conv, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyConvModelFloat(AbstractPyConvModel):
    """Conv with float synapse implementation."""
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float, precision=24)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    a_buf: np.ndarray = LavaPyType(np.ndarray, float)
    weight: np.ndarray = LavaPyType(np.ndarray, float)


@implements(proc=Conv, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyConvModelFixed(AbstractPyConvModel):
    """Conv with fixed point synapse implementation."""
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    a_buf: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    weight: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=8)

    def clamp_precision(self, x: np.ndarray) -> np.ndarray:
        return utils.signed_clamp(x, bits=24)
