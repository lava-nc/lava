# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from re import X
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
    weight_exp: float = LavaPyType(float, float)
    num_weight_bits: float = LavaPyType(float, float)
    sign_mode: float = LavaPyType(float, float)
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


@implements(proc=Dense, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyDenseModelBitAcc(PyLoihiProcessModel):
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
    weight_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=4)
    num_weight_bits: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    sign_mode: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=2)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def __init__(self, proc_params):
        super(PyDenseModelBitAcc, self).__init__(proc_params)
        # Flag to determine whether weights have already been scaled.
        self.weights_set = False

    def _set_wgts(self):
        wgt_vals = np.copy(self.weights)

        # Saturate the weights according to the sign_mode:
        # 0 : null
        # 1 : mixed
        # 2 : excitatory
        # 3 : inhibitory
        mixed_idx = np.equal(self.sign_mode, 1).astype(np.int32)
        excitatory_idx = np.equal(self.sign_mode, 2).astype(np.int32)
        inhibitory_idx = np.equal(self.sign_mode, 3).astype(np.int32)

        min_wgt = -(2**8) * (mixed_idx + inhibitory_idx)
        max_wgt = (2**8 - 1) * (mixed_idx + excitatory_idx)

        saturated_wgts = np.clip(wgt_vals, min_wgt, max_wgt)

        # Truncate least significant bits given sign_mode and num_wgt_bits.
        num_truncate_bits = 8 - self.num_weight_bits + mixed_idx

        truncated_wgts = np.left_shift(
            np.right_shift(saturated_wgts, num_truncate_bits), num_truncate_bits
        )

        wgt_vals = truncated_wgts.astype(np.int32)
        wgts_scaled = np.copy(wgt_vals)
        self.weights_set = True
        return wgts_scaled

    def run_spk(self):
        # Since this Process has no learning, weights are assumed to be static
        # and only require scaling on the first timestep of run_spk().
        if not self.weights_set:
            self.weights = self._set_wgts()
        # The a_out sent on a each timestep is a buffered value from dendritic
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
    weight_exp: float = LavaPyType(float, float)
    num_weight_bits: float = LavaPyType(float, float)
    sign_mode: float = LavaPyType(float, float)
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
    weight_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=4)
    num_weight_bits: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    sign_mode: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=2)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        # Flag to determine whether weights have already been scaled.
        self.weights_set = False

    def _set_wgts(self):
        wgt_vals = np.copy(self.weights)

        # Saturate the weights according to the sign_mode:
        # 0 : null
        # 1 : mixed
        # 2 : excitatory
        # 3 : inhibitory
        mixed_idx = np.equal(self.sign_mode, 1).astype(np.int32)
        excitatory_idx = np.equal(self.sign_mode, 2).astype(np.int32)
        inhibitory_idx = np.equal(self.sign_mode, 3).astype(np.int32)

        min_wgt = -(2**8) * (mixed_idx + inhibitory_idx)
        max_wgt = (2**8 - 1) * (mixed_idx + excitatory_idx)

        saturated_wgts = np.clip(wgt_vals, min_wgt, max_wgt)

        # Truncate least significant bits given sign_mode and num_wgt_bits.
        num_truncate_bits = 8 - self.num_weight_bits + mixed_idx

        truncated_wgts = np.left_shift(
            np.right_shift(saturated_wgts, num_truncate_bits), num_truncate_bits
        )

        wgt_vals = truncated_wgts.astype(np.int32)
        wgts_scaled = np.copy(wgt_vals)
        self.weights_set = True
        return wgts_scaled

    def run_spk(self):
        # Since this Process has no learning, weights are assumed to be static
        # and only require scaling on the first timestep of run_spk().
        if not self.weights_set:
            self.weights = self._set_wgts()
        # The a_out sent on a each timestep is a buffered value from dendritic
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
