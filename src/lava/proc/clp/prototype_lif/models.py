# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.model.py.neuron import LearningNeuronModelFixed
from lava.proc.clp.prototype_lif.process import PrototypeLIF

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.lif.models import AbstractPyLifModelFixed


@implements(proc=PrototypeLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class PrototypeLIFBitAcc(LearningNeuronModelFixed, AbstractPyLifModelFixed):
    """Implementation of Prototype Leaky-Integrate-and-Fire neural
    process bit-accurate with Loihi's hardware LIF dynamics,
    which means, it mimics Loihi behaviour bit-by-bit.

    Features of the PrototypeLIF neurons:
    - 3-factor learning: use the 3rd factor value as the individual learning
    rate. This is done by writing this value into y1 trace.
    - Use presence of the third factor as gating factor for learning via bAP
    signal. When a third factor is received, a bAP signal is generated and sent

    Precisions of state variables

    - du: unsigned 12-bit integer (0 to 4095)
    - dv: unsigned 12-bit integer (0 to 4095)
    - bias_mant: signed 13-bit integer (-4096 to 4095). Mantissa part of neuron
      bias.
    - bias_exp: unsigned 3-bit integer (0 to 7). Exponent part of neuron bias.
    - vth: unsigned 17-bit integer (0 to 131071).

    """
    # s_out is 24-bit graded value
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    reset_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=1)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.effective_vth = 0
        self.s_out_buff = np.zeros(proc_params["shape"])
        self.isthrscaled = False
        self.y1 = np.zeros(proc_params["shape"], dtype=np.int32)

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
        s_out_y1: sends the post-synaptic spike times.
        s_out_y2: sends the graded third-factor reward signal.
        """

        # Receive synaptic input and the 3rd factor input
        a_in_data = self.a_in.recv()
        a_3rd_factor_in = self.a_third_factor_in.recv().astype(np.int32)
        reset = self.reset_in.recv()

        # Scale the bias
        self.scale_bias()

        # Scale the threshold if is not already
        if not self.isthrscaled:
            self.scale_threshold()

        # Run sub-threshold dynamics
        self.subthr_dynamics(activation_in=a_in_data)

        # If a reset spike is received, reset both voltage and current
        if np.any(reset > 0):
            self.v[reset > 0] *= 0
            self.u[reset > 0] *= 0

        # Generate bAP signals the neurons that received its own id in the
        # 3rd factor channel. As all values of "a_3rd_factor_in" will be
        # same, we will check just the first one. Note that the id's sent in
        # channel start from one, not zero.
        s_out_bap_buff = np.zeros(shape=self.s_out_bap.shape, dtype=bool)
        if a_3rd_factor_in[0] != 0:
            s_out_bap_buff[a_3rd_factor_in[0] - 1] = True
        # Generate the output spikes
        self.s_out_buff = self.spiking_activation()

        # If there was any 3rd factor input to the population, then update y1
        # trace of those neurons to 127, the maximum value, because we are
        # doing one-shot learning. The y1 trace is used in learning rule as
        # the learning rate
        if s_out_bap_buff.any():
            self.y1 = s_out_bap_buff * 127
            self.s_out_buff = s_out_bap_buff.copy()

        # Send out the output & bAP spikes and update y1 trace
        self.s_out.send(self.s_out_buff)
        self.s_out_bap.send(s_out_bap_buff)
        self.s_out_y1.send(self.y1)

        # Reset voltage of spiked neurons to 0
        self.reset_voltage(spike_vector=self.s_out_buff)
