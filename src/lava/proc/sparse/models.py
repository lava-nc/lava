# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from scipy.sparse import csr_matrix, spmatrix, vstack, find
from lava.magma.core.model.py.connection import (
    LearningConnectionModelFloat,
    LearningConnectionModelBitApproximate,
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.sparse.process import Sparse, DelaySparse, LearningSparse
from lava.utils.weightutils import SignMode, determine_sign_mode,\
    truncate_weights, clip_weights


class AbstractPySparseModelFloat(PyLoihiProcessModel):
    """Implementation of Conn Process with Sparse synaptic connections in
    floating point precision. This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation.
    """

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    a_buff: np.ndarray = LavaPyType(np.ndarray, float)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons) in C-order (row major).
    weights: csr_matrix = LavaPyType(csr_matrix, float)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def run_spk(self):
        # The a_out sent on each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff)
        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
            self.a_buff = self.weights.dot(s_in)
        else:
            s_in = self.s_in.recv().astype(bool)
            # A1: return as flattend array
            self.a_buff = self.weights[:, s_in].sum(axis=1).A1


@implements(proc=Sparse, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PySparseModelFloat(AbstractPySparseModelFloat):
    pass


class AbstractPySparseModelBitAcc(PyLoihiProcessModel):
    """Implementation of Conn Process with Sparse synaptic connections that is
    bit-accurate with Loihi's hardware implementation of Sparse, which means,
    it reproduces Loihi behavior bit-by-bit.
    """

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    a_buff: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons) in C-order (row major).
    weights: np.ndarray = LavaPyType(csr_matrix, np.int32, precision=8)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def __init__(self, proc_params):
        super().__init__(proc_params)
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
            a_accum = self.weights[:, s_in].sum(axis=1).A1
        self.a_buff = (
            np.left_shift(a_accum, self.weight_exp)
            if self.weight_exp > 0
            else np.right_shift(a_accum, -self.weight_exp)
        )


@implements(proc=Sparse, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PySparseModelBitAcc(AbstractPySparseModelBitAcc):
    pass


@implements(proc=LearningSparse, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLearningSparseModelFloat(
        LearningConnectionModelFloat, AbstractPySparseModelFloat):
    """Implementation of Conn Process with Sparse synaptic connections in
    floating point precision. This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation.

    Warning: LearningSparse on CPU is not offering any memory usage benefits
    over using LearningDense.
    """

    # Overwrite dense PyTypes with sparse
    tag_1: csr_matrix = LavaPyType(csr_matrix, float)
    tag_2: csr_matrix = LavaPyType(csr_matrix, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)

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
            self.a_buff = self.weights[:, s_in].sum(axis=1).A1

        self.recv_traces(s_in)


@implements(proc=LearningSparse, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_approximate_loihi", "fixed_pt")
class PyLearningSparseModelBitApproximate(
        LearningConnectionModelBitApproximate, AbstractPySparseModelBitAcc):
    """Implementation of Conn Process with Sparse synaptic connections that
    uses similar constraints as Loihi's hardware implementation of sparse
    connectivity but does not reproduce Loihi bit-by-bit.

    Warning: LearningSparse on CPU is not offering any memory usage benefits
    over using LearningDense.
    """
    # Overwrite dense PyTypes with sparse
    tag_1: csr_matrix = LavaPyType(csr_matrix, int, precision=8)
    tag_2: csr_matrix = LavaPyType(csr_matrix, int, precision=6)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        # Flag to determine whether weights have already been scaled.
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
            a_accum = self.weights[:, s_in].sum(axis=1).A1

        self.a_buff = (
            np.left_shift(a_accum, self.weight_exp)
            if self.weight_exp > 0
            else np.right_shift(a_accum, -self.weight_exp)
        )

        self.recv_traces(s_in)


class AbstractPyDelaySparseModel(PyLoihiProcessModel):
    """Abstract Conn Process with Sparse synaptic connections which incorporates
    delays into the Conn Process.
    """
    weights: csr_matrix = None
    delays: csr_matrix = None
    a_buff: np.ndarray = None

    def calc_act(self, s_in) -> np.ndarray:
        """
        Calculate the activation matrix based on s_in by performing
        delay_wgts * s_in.
        """
        # First calculating the activations through delay_wgts * s_in
        # This matrix is then summed across each row to get the
        # activations to the output neurons for different delays.
        # This activation vector is reshaped to a matrix of the form
        # (n_flat_output_neurons * (max_delay + 1), n_flat_output_neurons)
        #  which is then transposed to get the activation matrix.
        return np.reshape(self.get_delay_wgts_mat(self.weights,
                                                  self.delays,
                                                  self.a_buff.shape[-1] - 1
                                                  ).dot(s_in),
                          (self.a_buff.shape[-1], self.weights.shape[0])).T

    @staticmethod
    def get_delay_wgts_mat(weights, delays, max_delay) -> spmatrix:
        """
        Create a matrix where the synaptic weights are separated
        by their corresponding delays. The first matrix contains all the
        weights, where the delay is equal to zero. The second matrix
        contains all the weights, where the delay is equal to one and so on.
        These matrices are then stacked together vertically.

        Returns 2D matrix of form
        (num_flat_output_neurons * max_delay + 1, num_flat_input_neurons) where
        delay_wgts[
            k * num_flat_output_neurons : (k + 1) * num_flat_output_neurons, :
        ]
        contains the weights for all connections with a delay equal to k.
        This allows for the updating of the activation buffer and updating
        weights.
        """
        # Can only start at 1, as delays==0 raises inefficiency warning
        if max_delay == 0:
            return weights

        weight_delay_from_1 = vstack([weights.multiply(delays == k)
                                      for k in range(1, max_delay + 1)])
        # Create weight matrix at delays == 0
        r, c, _ = find(delays)
        weight_delay_zeros = weights.copy()
        weight_delay_zeros[r, c] = 0
        weight_delay_zeros.eliminate_zeros()
        return vstack([weight_delay_zeros, weight_delay_from_1])

    def update_act(self, s_in):
        """
        Updates the activations for the connection.
        Clears first column of a_buff and rolls them to the last column.
        Finally, calculates the activations for the current time step and adds
        them to a_buff.
        This order of operations ensures that delays of 0 correspond to
        the next time step.
        """
        self.a_buff[:, 0] = 0
        self.a_buff = np.roll(self.a_buff, -1)
        self.a_buff += self.calc_act(s_in)


@implements(proc=DelaySparse, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyDelaySparseModelFloat(AbstractPyDelaySparseModel):
    """Implementation of Conn Process with Sparse synaptic connections in
    floating point precision. This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation. DelaySparse incorporates delays into the Conn
    Process.
    """
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    a_buff: np.ndarray = LavaPyType(np.ndarray, float)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons) in C-order (row major).
    weights: np.ndarray = LavaPyType(csr_matrix, float)
    # delays is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons) in C-order (row major).
    delays: np.ndarray = LavaPyType(csr_matrix, int)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def run_spk(self):
        # The a_out sent on each timestep is a buffered value from dendritic
        # accumulation at timestep t-1; this prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff[:, 0])
        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
        else:
            s_in = self.s_in.recv().astype(bool)
        self.update_act(s_in)


@implements(proc=DelaySparse, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyDelaySparseModelBitAcc(AbstractPyDelaySparseModel):
    """Implementation of Conn Process with Sparse synaptic connections that is
    bit-accurate with Loihi's hardware implementation of Sparse, which means,
    it mimics Loihi behaviour bit-by-bit. DelaySparse incorporates delays into
    the Conn Process. Loihi 2 has a maximum of 6 bits for delays, meaning a
    spike can be delayed by 0 to 63 time steps."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=16)
    a_buff: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons) in C-order (row major).
    weights: csr_matrix = LavaPyType(csr_matrix, np.int32, precision=8)
    delays: csr_matrix = LavaPyType(csr_matrix, np.int32, precision=6)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def __init__(self, proc_params):
        super().__init__(proc_params)
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

            # Check if delays are within Loihi 2 constraints
            if np.max(self.delays) > 63:
                raise ValueError("DelaySparse Process 'delays' expects values "
                                 f"between 0 and 63 for Loihi, got "
                                 f"{self.delays}.")

        # The a_out sent at each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff[:, 0])
        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
        else:
            s_in = self.s_in.recv().astype(bool)

        a_accum = self.calc_act(s_in)
        self.a_buff[:, 0] = 0
        self.a_buff = np.roll(self.a_buff, -1)
        self.a_buff += (
            np.left_shift(a_accum, self.weight_exp)
            if self.weight_exp > 0
            else np.right_shift(a_accum, -self.weight_exp)
        )
