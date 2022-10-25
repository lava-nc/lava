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
from lava.proc.dense.process import Dense, LearningDense, DenseDelay
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

@implements(proc=DenseDelay, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyDenseDelayModelFloat(PyLoihiProcessModel):
    """Implementation of Conn Process with Dense synaptic connections in
    floating point precision. This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation. DenseDelay incorporates delays into the Conn
    Process.
    """

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    a_buff: np.ndarray = LavaPyType(np.ndarray, float)
    # weights is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons) in C-order (row major).
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    # delays is a 2D matrix of form (num_flat_output_neurons,
    # num_flat_input_neurons) in C-order (row major).
    delays: np.ndarray = LavaPyType(np.ndarray, int)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def get_del_wgts(self):
        """
        Use self.weights and self.delays to create a matrix where the
        weights are separated by delay. Returns 2D matrix of form
        (num_flat_output_neurons * max_delay + 1, num_flat_input_neurons) where
        del_wgts[
            k * num_flat_output_neurons : (k + 1) * num_flat_output_neurons, :
        ]
        contains the weights for all connections with a delay equal to k.
        This allows for the updating of the activation buffer and updating
        weights.
        """
        return np.vstack([
            np.where(self.delays==k, self.weights, 0)
            for k in range(np.max(self.delays) + 1)
        ])

    def calc_act(self, s_in):
        """
        Calculate the activations by performing del_wgts * s_in. This matrix 
        is then summed across each row to get the activations to the output
        neurons for different delays. This activation vector is reshaped to a
        matrix of the form
        (n_flat_output_neurons * (max_delay + 1), n_flat_output_neurons)
        which is then transposed to get the activation matrix.
        """
        return np.reshape(
           np.sum(self.get_del_wgts() * s_in, axis=1),
           (np.max(self.delays) + 1, self.weights.shape[0])
        ).T

    def update_act(self, s_in):
        """
        Updates the activations for the connection.
        First, updates a_out with the first column of a_buff.
        Then clears these values from a_buff and rolls them to the last column.
        Finally, calculates the activations for the current time step and adds
        them to a_buff.
        This order of operations ensures that delays of 0 correspond to
        the next time step.
        """
        self.a_out = self.a_buff[:,0]
        self.a_buff[:,0] = 0
        self.a_buff = np.roll(self.a_buff, -1)
        self.a_buff += self.calc_act(s_in)

    def run_spk(self):
        # The a_out sent on a each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff[:,0])
        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
            self.update_act(s_in)
        else:
            s_in = self.s_in.recv().astype(bool)
            self.update_act(s_in)

            
