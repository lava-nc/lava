# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2022 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
from scipy.sparse import csr_matrix, find

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.utils.weightutils import (
    SignMode,
    optimize_weight_bits,
    truncate_weights,
    determine_sign_mode,
    clip_weights,
)

from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


DEBUG_INFO = False
DEBUG_INFO2 = False


class Syn(AbstractProcess):
    def __init__(
        self,
        *,
        weights: np.ndarray,
        name: ty.Optional[str] = None,
        num_message_bits: ty.Optional[int] = 0,
        log_config: ty.Optional[LogConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            weights=weights,
            num_message_bits=num_message_bits,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self._validate_weights(weights)
        shape = weights.shape

        # Ports
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))

        # print("shape: ", shape)

        # Variables
        self.weights = Var(shape=shape, init=weights)

    @staticmethod
    def _validate_weights(weights: np.ndarray) -> None:
        if len(np.shape(weights)) != 2:
            raise ValueError(
                "Dense Process 'weights' expects a 2D matrix, "
                f"got {weights}."
            )


@implements(proc=Syn, protocol=LoihiProtocol)
@requires(CPU)
class SynModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(
        PyInPort.VEC_DENSE, np.int64, precision=24
    )  # * makes it np.int64 due to the "Send data too large" problem
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int64, precision=24)
    weights: np.ndarray = LavaPyType(np.ndarray, np.int64, precision=8)

    def __init__(self, proc_params):
        super(SynModel, self).__init__(proc_params)
        self.synname = proc_params.get("synname", "")
        self.firsttime = True

        self.input_shape = None
        self.output_shape = None

    def init(self):
        weight_exp: int = self.proc_params.get("weight_exp", 0)
        num_weight_bits: int = self.proc_params.get("num_weight_bits", 8)
        sign_mode: SignMode = self.proc_params.get("sign_mode", None)
        weights: np.ndarray = self.weights

        input_shape = weights.shape[0]
        output_shape = weights.shape[1]
        self.input_shape = input_shape
        self.output_shape = output_shape

        if DEBUG_INFO:
            print("input_shape:", input_shape)
            print("===================")
            print("output_shape:", output_shape)
            print("===================")

        if DEBUG_INFO:
            print("weights:", weights.shape)
            print("===================")

        sign_mode = sign_mode or determine_sign_mode(weights)
        weights = clip_weights(weights, sign_mode, num_bits=8)
        weights = truncate_weights(weights, sign_mode, num_weight_bits)
        optimized_weights = optimize_weight_bits(
            weights=weights, sign_mode=sign_mode, loihi2=True
        )

        weights: np.ndarray = optimized_weights.weights
        weights_exp: int = optimized_weights.weight_exp + weight_exp

        if DEBUG_INFO:
            print("weights:", weights.shape)
            print("===================")
            print("weights_exp:", weights_exp)
            print("===================")

        num_weight_bits = optimized_weights.num_weight_bits

        sparse_weights = csr_matrix(weights)
        dst, src, wgt = find(
            sparse_weights
        )  # * Return the indices and values of the nonzero
        # elements of a matrix

        if DEBUG_INFO2:
            print(
                self.synname,
                "dst:",
                dst.shape,
                "     src: ",
                src.shape,
                "     wgt: ",
                wgt.shape,
            )
            print("===================")

        # Sort sparse weight in the order of input dimension
        idx = np.argsort(src)
        src = src[idx]
        dst = dst[idx]
        wgt = wgt[idx]
        self.wgt = wgt
        self.src = src
        self.dst = dst
        self.oup_data = np.zeros(self.output_shape)

        if DEBUG_INFO:
            print("sparse_weights:", sparse_weights.shape)
            print("===================")
            print("dst:", dst, "     src: ", src, "     wgt: ", wgt)
            print("===================")

    def run_spk(self):
        if self.firsttime:
            self.init()
            self.firsttime = False
        inp = np.zeros(self.s_in.shape)
        if self.s_in.probe():
            # print(self.synname, "recv")
            inp = self.s_in.recv()
        expanded_inp = inp[self.src]
        expanded_oup = expanded_inp * self.wgt
        for i in range(len(self.dst)):
            self.oup_data[self.dst[i]] += expanded_oup[i]

        output = np.zeros(self.output_shape)
        THRESH = 0.5
        for i in range(len(self.oup_data)):
            if self.oup_data[i] > THRESH:
                output[i] = 1
                self.oup_data[i] -= THRESH
        if DEBUG_INFO:
            print(self.synname, "run_spk")
            print(self.synname, "wgt", self.wgt.shape)
        # print("a_out sending ", output.shape)
        self.a_out.send(output)
