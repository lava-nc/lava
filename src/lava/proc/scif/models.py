# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.lfsr_model import adv_lfsr_nbits
from lava.proc.scif.process import SCIF


@implements(proc=SCIF, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PySCIFModelFixed(PyLoihiProcessModel):
    """Fixed point implementation of Stochastic Constraint Integrate and
    Fire (SCIF) neuron.
    """
    a_in = LavaPyType(PyInPort.VEC_DENSE, int, precision=8)
    s_sig_out = LavaPyType(PyOutPort.VEC_DENSE, int, precision=8)
    s_wta_out = LavaPyType(PyOutPort.VEC_DENSE, int, precision=8)

    u: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    beta: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    bias: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    theta: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    neg_tau_ref: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    enable_noise: np.ndarray = LavaPyType(np.ndarray, int, precision=1)

    def run_spk(self) -> None:
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        # Define spike vectors
        s_sig = np.zeros_like(self.v)
        s_wta = np.zeros_like(self.v)

        # Saturated add the incoming activation
        self.u += a_in_data
        self.u[self.u > 2 ** 23 - 1] = 2 ** 23 - 1
        self.u[self.u < -2 ** 23] = -2 ** 23

        # Populate the buffer for local computation
        lsb = self.beta.copy()
        lsb &= 3
        self.beta <<= 2
        self.beta[self.beta >= 256] = 0  # Overflow 8-bit unsigned beta to 0

        # Gather spike and unsatisfied indices for summation axons
        sig_unsat_idx = np.where(lsb == 2)
        sig_spk_idx = np.where(np.logical_and(lsb == 1, self.u == 0))

        # First set of unsatisfied WTA indices based on beta and u
        wta_unsat_idx = np.where(np.logical_and(lsb == 1, self.u < 0))

        # Reset voltages of unsatisfied WTA
        self.v[wta_unsat_idx] = 0

        # Assign sigma spikes (+/- 1)
        s_sig[sig_unsat_idx] = -1
        s_sig[sig_spk_idx] = 1

        # Determine neurons under refractory and not refractory
        rfct_idx = np.where(self.v < 0)  # indices of neurons in refractory
        not_rfct_idx = np.where(self.v >= 0)  # neurons not in refractory

        # Split/fork state variables u, v, beta
        v_in_rfct = self.v[rfct_idx]  # voltages in refractory
        u_in_rfct = self.u[rfct_idx]  # currents in refractory
        beta_in_rfct = self.beta[rfct_idx]  # beta in refractory
        v_to_intg = self.v[not_rfct_idx]  # voltages to be integrated
        u_to_intg = self.u[not_rfct_idx]  # currents to be integrated
        beta_to_intg = self.beta[not_rfct_idx]  # beta to be integrated
        bias_to_intg = self.bias[not_rfct_idx]  # bias to be integrated

        # Integration of constraints
        #  ToDo: Choosing a 16-bit signed random integer. For bit-accuracy,
        #   need to replace it with Loihi-conformant LFSR function
        # If noise is enabled, choose a 16-bit signed random integer,
        # else choose zeros
        lfsr = np.zeros_like(v_to_intg)
        if lfsr.size > 0:
            rand_nums = \
                np.random.randint(-2 ** 15, 2 ** 15 - 1,
                                  size=np.count_nonzero(self.enable_noise == 1))
            lfsr[self.enable_noise == 1] = rand_nums

        lfsr = np.right_shift(lfsr, 1)
        v_to_intg = v_to_intg + lfsr + u_to_intg + bias_to_intg
        v_to_intg[v_to_intg > 2 ** 23 - 1] = 2 ** 23 - 1  # Saturate at max
        v_to_intg[v_to_intg < 0] = 0  # Remove negatives

        # WTA spike indices when threshold is exceeded
        wta_spk_idx = np.where(v_to_intg >= self.theta)  # Exceeds threshold

        # Spiking neuron voltages go in refractory
        v_to_intg[wta_spk_idx] = self.neg_tau_ref  # Post spk refractory
        beta_to_intg[wta_spk_idx] |= 1
        s_wta[wta_spk_idx] = 1  # issue +1 WTA spikes

        # Refractory dynamics
        v_in_rfct += 1  # voltage increments by 1 every step
        beta_in_rfct |= 3
        # Second set of unsatisfied WTA indices based on v and u in refractory
        wta_unsat_idx_2 = np.where(np.logical_or(v_in_rfct == 0, u_in_rfct < 0))

        # Reset voltage of unsatisfied WTA in refractory
        v_in_rfct[wta_unsat_idx_2] = 0
        beta_in_rfct[wta_unsat_idx_2] &= 2
        s_wta[wta_unsat_idx] = -1
        s_wta[wta_unsat_idx_2] = -1

        # Assign all temporary states to state Vars
        self.v[rfct_idx] = v_in_rfct
        self.v[not_rfct_idx] = v_to_intg
        self.u[rfct_idx] = u_in_rfct
        self.u[not_rfct_idx] = u_to_intg
        self.beta[rfct_idx] = beta_in_rfct
        self.beta[not_rfct_idx] = beta_to_intg

        # Send out spikes
        self.s_sig_out.send(s_sig)
        self.s_wta_out.send(s_wta)
