# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.conv_in_time.process import ConvInTime
from lava.proc.conv import utils


class AbstractPyConvInTimeModel(PyLoihiProcessModel):
    """Abstract Conn In Time Process with Dense synaptic connections
    which incorporates delays into the Conv Process.
    """
    weights: np.ndarray = None
    a_buff: np.ndarray = None
    kernel_size: int = None

    num_message_bits: np.ndarray = LavaPyType(np.ndarray, np.int8, precision=5)

    def calc_act(self, s_in) -> np.ndarray:
        """
        Calculate the activation buff by inverse the order in
        the kernel. Taking k=3 as an example, the a_buff will be
        weights[2] * s_in, weights[1] * s_in, weights[0] * s_in
        """

        # The change of the shape is shown below:
        # sum([K, n_out, n_in] * [n_in,], axis=-1) = [K, n_out] -> [n_out, K]
        kernel_size = self.weights.shape[0]
        for i in range(kernel_size):
            self.a_buff[:, i] += np.sum(
                self.weights[kernel_size - i - 1] * s_in, axis=-1).T

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
        self.calc_act(s_in)

    def run_spk(self):
        # The a_out sent on a each timestep is a buffered value from dendritic
        # accumulation at timestep t-1. This prevents deadlocking in
        # networks with recurrent connectivity structures.
        self.a_out.send(self.a_buff[:, 0])
        if self.num_message_bits.item() > 0:
            s_in = self.s_in.recv()
        else:
            s_in = self.s_in.recv().astype(bool)
        self.update_act(s_in)


@implements(proc=ConvInTime, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyConvInTimeFloat(AbstractPyConvInTimeModel):
    """Implementation of Conn In Time Process with Dense synaptic connections in
    floating point precision. This short and simple ProcessModel can be used
    for quick algorithmic prototyping, without engaging with the nuances of a
    fixed point implementation. DelayDense incorporates delays into the Conn
    Process.
    """
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    a_buff: np.ndarray = LavaPyType(np.ndarray, float)
    # The weights is a 3D matrix of form (kernel_size,
    # num_flat_output_neurons, num_flat_input_neurons) in C-order (row major).
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)


@implements(proc=ConvInTime, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyConvInTimeFixed(AbstractPyConvInTimeModel):
    """Conv In Time with fixed point synapse implementation."""
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    a_buff: np.ndarray = LavaPyType(np.ndarray, float)
    # The weights is a 3D matrix of form (kernel_size,
    # num_flat_output_neurons, num_flat_input_neurons) in C-order (row major).
    weights: np.ndarray = LavaPyType(np.ndarray, float)
    num_message_bits: np.ndarray = LavaPyType(np.ndarray, int, precision=5)

    def clamp_precision(self, x: np.ndarray) -> np.ndarray:
        return utils.signed_clamp(x, bits=24)
