# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.model.py.neuron import (
    LearningNeuronModelFloat,
    LearningNeuronModelFixed,
)
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.lif.process import (LIF, LIFReset, TernaryLIF, LearningLIF,
                                   LIFRefractory)


class AbstractPyLifModelFloat(PyLoihiProcessModel):
    """Abstract implementation of floating point precision
    leaky-integrate-and-fire neuron model.

    Specific implementations inherit from here.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = None  # This will be an OutPort of different LavaPyTypes
    u: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, float)
    du: float = LavaPyType(float, float)
    dv: float = LavaPyType(float, float)

    def spiking_activation(self):
        """Abstract method to define the activation function that determines
        how spikes are generated.
        """
        raise NotImplementedError(
            "spiking activation() cannot be called from "
            "an abstract ProcessModel"
        )

    def subthr_dynamics(self, activation_in: np.ndarray):
        """Common sub-threshold dynamics of current and voltage variables for
        all LIF models. This is where the 'leaky integration' happens.
        """
        self.u[:] = self.u * (1 - self.du)
        self.u[:] += activation_in
        self.v[:] = self.v * (1 - self.dv) + self.u + self.bias_mant

    def reset_voltage(self, spike_vector: np.ndarray):
        """Voltage reset behaviour. This can differ for different neuron
        models."""
        self.v[spike_vector] = 0

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        super().run_spk()
        a_in_data = self.a_in.recv()

        self.subthr_dynamics(activation_in=a_in_data)
        self.s_out_buff = self.spiking_activation()
        self.reset_voltage(spike_vector=self.s_out_buff)
        self.s_out.send(self.s_out_buff)


class AbstractPyLifModelFixed(PyLoihiProcessModel):
    """Abstract implementation of fixed point precision
    leaky-integrate-and-fire neuron model. Implementations like those
    bit-accurate with Loihi hardware inherit from here.
    """

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int16, precision=16)
    s_out: None  # This will be an OutPort of different LavaPyTypes
    u: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    du: int = LavaPyType(int, np.uint16, precision=12)
    dv: int = LavaPyType(int, np.uint16, precision=12)
    bias_mant: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=13)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=3)

    def __init__(self, proc_params):
        super(AbstractPyLifModelFixed, self).__init__(proc_params)
        # ds_offset and dm_offset are 1-bit registers in Loihi 1, which are
        # added to du and dv variables to compute effective decay constants
        # for current and voltage, respectively. They enable setting decay
        # constant values to exact 4096 = 2**12. Without them, the range of
        # 12-bit unsigned du and dv is 0 to 4095.
        self.ds_offset = 1
        self.dm_offset = 0
        self.isbiasscaled = False
        self.isthrscaled = False
        self.effective_bias = 0
        # Let's define some bit-widths from Loihi
        # State variables u and v are 24-bits wide
        self.uv_bitwidth = 24
        self.max_uv_val = 2 ** (self.uv_bitwidth - 1)
        # Decays need an MSB alignment with 12-bits
        self.decay_shift = 12
        self.decay_unity = 2**self.decay_shift
        # Threshold and incoming activation are MSB-aligned using 6-bits
        self.vth_shift = 6
        self.act_shift = 6

    def scale_bias(self):
        """Scale bias with bias exponent by taking into account sign of the
        exponent.
        """
        # Create local copy of bias_mant with promoted dtype to prevent
        # overflow when applying shift of bias_exp.
        bias_mant = self.bias_mant.copy().astype(np.int32)
        self.effective_bias = np.where(
            self.bias_exp >= 0,
            np.left_shift(bias_mant, self.bias_exp),
            np.right_shift(bias_mant, -self.bias_exp),
        )
        self.isbiasscaled = True

    def scale_threshold(self):
        """Placeholder method for scaling threshold(s)."""
        raise NotImplementedError(
            "spiking activation() cannot be called from "
            "an abstract ProcessModel"
        )

    def spiking_activation(self):
        """Placeholder method to specify spiking behaviour of a LIF neuron."""
        raise NotImplementedError(
            "spiking activation() cannot be called from "
            "an abstract ProcessModel"
        )

    def subthr_dynamics(self, activation_in: np.ndarray):
        """Common sub-threshold dynamics of current and voltage variables for
        all LIF models. This is where the 'leaky integration' happens.
        """

        # Update current
        # --------------
        decay_const_u = self.du + self.ds_offset
        # Below, u is promoted to int64 to avoid overflow of the product
        # between u and decay constant beyond int32. Subsequent right shift by
        # 12 brings us back within 24-bits (and hence, within 32-bits)
        decayed_curr = np.int64(self.u) * (self.decay_unity - decay_const_u)
        decayed_curr = np.sign(decayed_curr) * np.right_shift(
            np.abs(decayed_curr), self.decay_shift
        )
        decayed_curr = np.int32(decayed_curr)
        # Hardware left-shifts synaptic input for MSB alignment
        activation_in = np.left_shift(activation_in, self.act_shift)
        # Add synptic input to decayed current
        decayed_curr += activation_in
        # Check if value of current is within bounds of 24-bit. Overflows are
        # handled by wrapping around modulo 2 ** 23. E.g., (2 ** 23) + k
        # becomes k and -(2**23 + k) becomes -k
        wrapped_curr = np.where(
            decayed_curr > self.max_uv_val,
            decayed_curr - 2 * self.max_uv_val,
            decayed_curr,
        )
        wrapped_curr = np.where(
            wrapped_curr <= -self.max_uv_val,
            decayed_curr + 2 * self.max_uv_val,
            wrapped_curr,
        )
        self.u[:] = wrapped_curr
        # Update voltage
        # --------------
        decay_const_v = self.dv + self.dm_offset

        neg_voltage_limit = -np.int32(self.max_uv_val) + 1
        pos_voltage_limit = np.int32(self.max_uv_val) - 1
        # Decaying voltage similar to current. See the comment above to
        # understand the need for each of the operations below.
        decayed_volt = np.int64(self.v) * (self.decay_unity - decay_const_v)
        decayed_volt = np.sign(decayed_volt) * np.right_shift(
            np.abs(decayed_volt), self.decay_shift
        )
        decayed_volt = np.int32(decayed_volt)
        updated_volt = decayed_volt + self.u + self.effective_bias
        self.v[:] = np.clip(updated_volt, neg_voltage_limit, pos_voltage_limit)

    def reset_voltage(self, spike_vector: np.ndarray):
        """Voltage reset behaviour. This can differ for different neuron
        models.
        """
        self.v[spike_vector] = 0

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        self.scale_bias()
        # # Compute effective bias and threshold only once, not every time-step
        # if not self.isbiasscaled:
        #     self.scale_bias()

        if not self.isthrscaled:
            self.scale_threshold()

        self.subthr_dynamics(activation_in=a_in_data)

        self.s_out_buff = self.spiking_activation()

        # Reset voltage of spiked neurons to 0
        self.reset_voltage(spike_vector=self.s_out_buff)
        self.s_out.send(self.s_out_buff)


@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLifModelFloat(AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural process in floating
    point precision. This short and simple ProcessModel can be used for quick
    algorithmic prototyping, without engaging with the nuances of a fixed
    point implementation.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return self.v > self.vth


@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyLifModelBitAcc(AbstractPyLifModelFixed):
    """Implementation of Leaky-Integrate-and-Fire neural process bit-accurate
    with Loihi's hardware LIF dynamics, which means, it mimics Loihi
    behaviour bit-by-bit.

    Currently missing features (compared to Loihi 1 hardware):

    - refractory period after spiking
    - axonal delays

    Precisions of state variables

    - du: unsigned 12-bit integer (0 to 4095)
    - dv: unsigned 12-bit integer (0 to 4095)
    - bias_mant: signed 13-bit integer (-4096 to 4095). Mantissa part of neuron
      bias.
    - bias_exp: unsigned 3-bit integer (0 to 7). Exponent part of neuron bias.
    - vth: unsigned 17-bit integer (0 to 131071).

    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super(PyLifModelBitAcc, self).__init__(proc_params)
        self.effective_vth = 0

    def scale_threshold(self):
        """Scale threshold according to the way Loihi hardware scales it. In
        Loihi hardware, threshold is left-shifted by 6-bits to MSB-align it
        with other state variables of higher precision.
        """
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)
        self.isthrscaled = True

    def spiking_activation(self):
        """Spike when voltage exceeds threshold."""
        return self.v > self.effective_vth


@implements(proc=TernaryLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyTernLifModelFloat(AbstractPyLifModelFloat):
    """Implementation of Ternary Leaky-Integrate-and-Fire neural process in
    floating point precision. This ProcessModel builds upon the floating
    point ProcessModel for LIF by adding upper and lower threshold voltages.
    """

    # Spikes now become 2-bit signed floating point numbers
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float, precision=2)
    vth_hi: float = LavaPyType(float, float)
    vth_lo: float = LavaPyType(float, float)

    def spiking_activation(self):
        """Spiking activation for T-LIF: -1 spikes below lower threshold,
        +1 spikes above upper threshold.
        """
        return (-1) * (self.v < self.vth_lo) + (self.v > self.vth_hi)

    def reset_voltage(self, spike_vector: np.ndarray):
        """Reset voltage of all spiking neurons to 0."""
        self.v[spike_vector != 0] = 0  # Reset voltage to 0 wherever we spiked


@implements(proc=TernaryLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyTernLifModelFixed(AbstractPyLifModelFixed):
    """Implementation of Ternary Leaky-Integrate-and-Fire neural process
    with fixed point precision.

    See Also
    --------
    lava.proc.lif.models.PyLifModelBitAcc: Bit-Accurate LIF neuron model
    """

    # Spikes now become 2-bit signed integers
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=2)
    vth_hi: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    vth_lo: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def __init__(self, proc_params):
        super(PyTernLifModelFixed, self).__init__(proc_params)
        self.effective_vth_hi = 0
        self.effective_vth_lo = 0

    def scale_threshold(self):
        self.effective_vth_hi = np.left_shift(self.vth_hi, self.vth_shift)
        self.effective_vth_lo = np.left_shift(self.vth_lo, self.vth_shift)
        self.isthrscaled = True

    def spiking_activation(self):
        # Spike when exceeds threshold
        # ----------------------------
        neg_spikes = self.v < self.effective_vth_lo
        pos_spikes = self.v > self.effective_vth_hi
        return (-1) * neg_spikes + pos_spikes

    def reset_voltage(self, spike_vector: np.ndarray):
        """Reset voltage of all spiking neurons to 0."""
        self.v[spike_vector != 0] = 0  # Reset voltage to 0 wherever we spiked


@implements(proc=LIFReset, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLifResetModelFloat(AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural process with reset
    in floating point precision.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super(PyLifResetModelFloat, self).__init__(proc_params)
        self.reset_interval = proc_params["reset_interval"]
        self.reset_offset = (proc_params["reset_offset"]) % self.reset_interval

    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return self.v > self.vth

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        if (self.time_step % self.reset_interval) == self.reset_offset:
            self.u *= 0
            self.v *= 0

        self.subthr_dynamics(activation_in=a_in_data)

        s_out = self.spiking_activation()

        # Reset voltage of spiked neurons to 0
        self.reset_voltage(spike_vector=s_out)
        self.s_out.send(s_out)


@implements(proc=LIFReset, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyLifResetModelBitAcc(AbstractPyLifModelFixed):
    """Implementation of Leaky-Integrate-and-Fire neural process with reset
    bit-accurate with Loihi's hardware LIF dynamics, which means, it mimics
    Loihi behaviour.

    Precisions of state variables

    - du: unsigned 12-bit integer (0 to 4095)
    - dv: unsigned 12-bit integer (0 to 4095)
    - bias_mant: signed 13-bit integer (-4096 to 4095). Mantissa part of neuron
      bias.
    - bias_exp: unsigned 3-bit integer (0 to 7). Exponent part of neuron bias.
    - vth: unsigned 17-bit integer (0 to 131071).

    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super(PyLifResetModelBitAcc, self).__init__(proc_params)
        self.effective_vth = 0
        self.reset_interval = proc_params["reset_interval"]
        self.reset_offset = (proc_params["reset_offset"]) % self.reset_interval

    def scale_threshold(self):
        """Scale threshold according to the way Loihi hardware scales it. In
        Loihi hardware, threshold is left-shifted by 6-bits to MSB-align it
        with other state variables of higher precision.
        """
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)
        self.isthrscaled = True

    def spiking_activation(self):
        """Spike when voltage exceeds threshold."""
        return self.v > self.effective_vth

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        if (self.time_step % self.reset_interval) == self.reset_offset:
            self.u *= 0
            self.v *= 0

        self.scale_bias()

        if not self.isthrscaled:
            self.scale_threshold()

        self.subthr_dynamics(activation_in=a_in_data)

        s_out = self.spiking_activation()

        # Reset voltage of spiked neurons to 0
        self.reset_voltage(spike_vector=s_out)
        self.s_out.send(s_out)


@implements(proc=LIFRefractory, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLifRefractoryModelFloat(AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural process with
    refractory period in floating point precision.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)
    refractory_period_end: np.ndarray = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params):
        super(PyLifRefractoryModelFloat, self).__init__(proc_params)
        self.refractory_period = proc_params["refractory_period"]

    def spiking_activation(self):
        """Spiking activation function for LIF Refractory."""
        return self.v > self.vth

    def subthr_dynamics(self, activation_in: np.ndarray):
        """Sub-threshold dynamics of current and voltage variables for
        all refractory LIF models. This is where the 'leaky integration'
        happens.
        """
        self.u[:] = self.u * (1 - self.du)
        self.u[:] += activation_in
        non_refractory = self.refractory_period_end < self.time_step
        self.v[non_refractory] = self.v[non_refractory] * (1 - self.dv) + (
            self.u[non_refractory] + self.bias_mant[non_refractory])

    def process_spikes(self, spike_vector: np.ndarray):
        self.refractory_period_end[spike_vector] = (self.time_step
                                                    + self.refractory_period)
        super().reset_voltage(spike_vector)

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        self.subthr_dynamics(activation_in=a_in_data)

        s_out = self.spiking_activation()

        # Reset voltage of spiked neurons to 0
        self.process_spikes(spike_vector=s_out)
        self.s_out.send(s_out)


@implements(proc=LearningLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PyLearningLIFModelFixed(
    LearningNeuronModelFixed, AbstractPyLifModelFixed
):
    """Implementation of Leaky-Integrate-and-Fire neural
    process in fixed point precision with learning enabled.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.effective_vth = 0

    def scale_threshold(self):
        """Scale threshold according to the way Loihi hardware scales it. In
        Loihi hardware, threshold is left-shifted by 6-bits to MSB-align it
        with other state variables of higher precision.
        """
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)
        self.isthrscaled = True

    def spiking_activation(self):
        """Spike when voltage exceeds threshold."""
        return self.v > self.effective_vth

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        """
        super().run_spk()


@implements(proc=LearningLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLearningLifModelFloat(LearningNeuronModelFloat,
                              AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural process in floating
    point precision with learning enabled.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return self.v > self.vth

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        """
        super().run_spk()
