# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.lif.process import LIF


@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyLifModel1(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, np.float)
    v: np.ndarray = LavaPyType(np.ndarray, np.float)
    bias: np.ndarray = LavaPyType(np.ndarray, np.float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, np.float)
    du: float = LavaPyType(float, np.float)
    dv: float = LavaPyType(float, np.float)
    vth: float = LavaPyType(float, np.float)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.u[:] = self.u * (1 - self.du)
        self.u[:] += a_in_data
        bias = self.bias * (2**self.bias_exp)
        self.v[:] = self.v * (1 - self.dv) + self.u + bias
        s_out = self.v >= self.vth
        self.v[s_out] = 0  # Reset voltage to 0
        self.s_out.send(s_out)


@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
@tag('bit_accurate_loihi', 'fixed_pt')
class PyLifModel2(PyLoihiProcessModel):
    """Implementation of Leaky-Integrate-and-Fire neural process bit-accurate
    with Loihi's hardware LIF dynamics, which means, it mimics Loihi
    behaviour bit-by-bit.

    Currently missing features (compared to Loihi hardware):
        - refractory period after spiking
        - axonal delays

    Precisions of state variables
    -----------------------------
    du: unsigned 12-bit integer (0 to 4095)
    dv: unsigned 12-bit integer (0 to 4095)
    bias: signed 13-bit integer (-4096 to 4095). Mantissa part of neuron bias.
    bias_exp: unsigned 3-bit integer (0 to 7). Exponent part of neuron bias.
    vth: unsigned 17-bit integer (0 to 131071)
    """
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    du: int = LavaPyType(int, np.uint16, precision=12)
    dv: int = LavaPyType(int, np.uint16, precision=12)
    bias: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=13)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=3)
    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=17)

    def __init__(self):
        super(PyLifModel2, self).__init__()
        self.ds_offset = 1  # hardware specific 1-bit value added to current
        # decay
        self.dm_offset = 0  # hardware specific 1-bit value added to voltage
        # decay

    def run_spk(self):
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        # Update current
        # --------------
        decay_const_u = self.du + self.ds_offset
        # Below, u is promoted to int64 to avoid over flow of the product
        # between u and decay constant beyond int32. Subsequent right shift by
        # 12 brings us back within 24-bits (and hence, within 32-bits)
        decayed_curr = np.right_shift(np.int64(self.u) * (np.left_shift(1, 12)
                                                          - decay_const_u), 12,
                                      dtype=np.int32)
        # Hardware left-shifts synpatic input for MSB alignment
        a_in_data = np.left_shift(a_in_data, 6)
        # Add synptic input to decayed current
        decayed_curr += a_in_data
        # Check if value of current is within bounds
        neg_current_limit = -np.left_shift(1, 23, dtype=np.int32)
        pos_current_limit = np.left_shift(1, 23, dtype=np.int32) - 1
        clipped_curr = np.clip(decayed_curr, neg_current_limit,
                               pos_current_limit)
        self.u[:] = clipped_curr
        # Update voltage
        # --------------
        decay_const_v = self.dv + self.dm_offset
        neg_voltage_limit = -np.left_shift(1, 23, dtype=np.int32) + 1
        pos_voltage_limit = np.left_shift(1, 23, dtype=np.int32) - 1
        # Decaying voltage similar to current. See the comment above to
        # understand the need for each of the operations below.
        decayed_volt = np.right_shift(np.int64(self.v) * (np.left_shift(1, 12)
                                                          - decay_const_v), 12,
                                      dtype=np.int32)
        bias = np.left_shift(self.bias, self.bias_exp, dtype=np.int32)
        updated_volt = decayed_volt + self.u + bias
        self.v[:] = np.clip(updated_volt, neg_voltage_limit, pos_voltage_limit)

        # Spike when exceeds threshold
        # ----------------------------
        # In Loihi, user specified threshold is just the mantissa, with a
        # constant exponent of 6
        vth = np.left_shift(self.vth, 6, dtype=np.int32)
        s_out = self.v >= vth
        # Reset voltage of spiked neurons to 0
        self.v[s_out] = 0
        self.s_out.send(s_out)
