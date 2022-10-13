# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.model.py.connection import (
    ConnectionModelFloat,
    ConnectionModelBitApproximate,
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.dense.process import Dense, LearningDense
from lava.utils.weightutils import SignMode, determine_sign_mode,\
    truncate_weights, clip_weights


@implements(proc=Dense, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyDenseModelFloat(PyLoihiProcessModel):
    """Implementation of Conn Process with Dense synaptic connections in
    floating point precision. This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation.
    """

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    a_buff: np.ndarray = LavaPyType(np.ndarray, float)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons)in C-order (row major).
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def run_spk(self):
        # The a_out sent on a each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff)
        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
            self.a_buff = self.weights.dot(s_in)
        else:
            s_in = self.s_in.recv().astype(bool)
            self.a_buff = self.weights[:, s_in].sum(axis=1)

        if self._learning_rule is not None:
            self._record_pre_spike_times(s_in)

        super().run_spk()


@implements(proc=Dense, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyDenseModelBitAcc(PyLoihiProcessModel):
    """Implementation of Conn Process with Dense synaptic connections that is
    bit-accurate with Loihi's hardware implementation of Dense, which means,
    it mimics Loihi behavior bit-by-bit.
    """

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    a_buff: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons) in C-order (row major).
    weights: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=8)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def __init__(self, proc_params):
        super(PyDenseModelBitAcc, self).__init__(proc_params)
        # Flag to determine whether weights have already been scaled.
        self.weights_set = False

    def run_spk(self):
        self.weight_exp: int = self.proc_params.get("weight_exp", 0)

        # Since this Process has no learning, weights are assumed to be static
        # and only require scaling on the first timestep of run_spk().
        if not self.weights_set:
            num_weight_bits: int = self.proc_params.get("num_weight_bits", 8)
            sign_mode: SignMode = self.proc_params.get("sign_mode") \
                or determine_sign_mode(self.weights)

            self.weights = clip_weights(self.weights, sign_mode, num_bits=8)
            self.weights = truncate_weights(self.weights,
                                            sign_mode,
                                            num_weight_bits)
            self.weights_set = True

        # The a_out sent at each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff)
        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
            a_accum = self.weights.dot(s_in)
        else:
            s_in = self.s_in.recv().astype(bool)
            a_accum = self.weights[:, s_in].sum(axis=1)
        self.a_buff = (
            np.left_shift(a_accum, self.weight_exp)
            if self.weight_exp > 0
            else np.right_shift(a_accum, -self.weight_exp)
        )


@implements(proc=LearningDense, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLearningDenseModelFloat(ConnectionModelFloat):
    """Implementation of Conn Process with Dense synaptic connections in
    floating point precision. This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation.
    """

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    a_buff: np.ndarray = LavaPyType(np.ndarray, float)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons)in C-order (row major).
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def run_spk(self):
        # The a_out sent at each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff)
        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
            self.a_buff = self.weights.dot(s_in)
        else:
            s_in = self.s_in.recv().astype(bool)
            self.a_buff = self.weights[:, s_in].sum(axis=1)

        if self._learning_rule is not None:
            self._record_pre_spike_times(s_in)

        super().run_spk()


@implements(proc=LearningDense, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_approximate_loihi", "fixed_pt")
class PyLearningDenseModelBitApproximate(ConnectionModelBitApproximate):
    """Implementation of Conn Process with Dense synaptic connections that is
    bit-accurate with Loihi's hardware implementation of Dense, which means,
    it mimics Loihi behaviour bit-by-bit.
    """

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    a_buff: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons) in C-order (row major).
    weights: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=8)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        # Flag to determine whether weights have already been scaled.
        self.weights_set = False
        self.num_weight_bits: int = self.proc_params.get("num_weight_bits", 8)

    def run_spk(self):
        self.weight_exp: int = self.proc_params.get("weight_exp", 0)

        # Since this Process has no learning, weights are assumed to be static
        # and only require scaling on the first timestep of run_spk().
        if not self.weights_set:
            self.weights = truncate_weights(
                self.weights,
                sign_mode=self.sign_mode,
                num_weight_bits=self.num_weight_bits
            )
            self.weights_set = True

        # The a_out sent at each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff)
        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
            a_accum = self.weights.dot(s_in)
        else:
            s_in = self.s_in.recv().astype(bool)
            a_accum = self.weights[:, s_in].sum(axis=1)

        self.a_buff = (
            np.left_shift(a_accum, self.weight_exp)
            if self.weight_exp > 0
            else np.right_shift(a_accum, -self.weight_exp)
        )

        if self._learning_rule is not None:
            self._record_pre_spike_times(s_in)

        super().run_spk()
