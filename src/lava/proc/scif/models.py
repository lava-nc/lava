# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from abc import abstractmethod

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.scif.process import CspScif, QuboScif


class AbstractPyModelScifFixed(PyLoihiProcessModel):
    """Abstract implementation of fixed point precision
    stochastic constraint integrate-and-fire neuron model. Implementations
    such as those bit-accurate with Loihi hardware inherit from here.
    """

    a_in = LavaPyType(PyInPort.VEC_DENSE, int, precision=8)
    s_sig_out = LavaPyType(PyOutPort.VEC_DENSE, int, precision=8)
    s_wta_out = LavaPyType(PyOutPort.VEC_DENSE, int, precision=8)

    cnstr_intg: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    state: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    spk_hist: np.ndarray = LavaPyType(np.ndarray, int, precision=8)

    step_size: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    theta: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    neg_tau_ref: np.ndarray = LavaPyType(np.ndarray, int, precision=24)
    noise_ampl: np.ndarray = LavaPyType(np.ndarray, int, precision=1)

    def __init__(self, proc_params):
        super(AbstractPyModelScifFixed, self).__init__(proc_params)
        self.a_in_data = np.zeros(proc_params['shape'])

    def _prng(self, intg_idx_):
        """Pseudo-random number generator
        """

        # ToDo: Choosing a 16-bit signed random integer. For bit-accuracy,
        #   need to replace it with Loihi-conformant LFSR function
        prand = np.zeros(shape=(len(intg_idx_),))
        if prand.size > 0:
            rand_nums = \
                np.random.randint(-2 ** 15, 2 ** 15 - 1, size=prand.size)
            # Assign random numbers only to neurons, for which noise is enabled
            prand = rand_nums * self.noise_ampl[intg_idx_]
        return prand

    def _update_buffers(self):
        # !! Side effect: Changes self.beta !!

        # Populate the buffer for local computation
        spk_hist_buffer = self.spk_hist.copy()
        spk_hist_buffer &= 3
        self.spk_hist <<= 2
        # Overflow 8-bit unsigned beta to 0
        self.spk_hist[self.spk_hist >= 256] = 0

        return spk_hist_buffer

    def _integration_dynamics(self, intg_idx):

        state_to_intg = self.state[intg_idx]  # voltages to be integrated
        cnstr_to_intg = self.cnstr_intg[intg_idx]  # currents to be integrated
        spk_hist_to_intg = self.spk_hist[intg_idx]  # beta to be integrated
        step_size_to_intg = self.step_size[intg_idx]  # bias to be integrated

        lfsr = self._prng(intg_idx_=intg_idx)

        state_to_intg = state_to_intg + lfsr + cnstr_to_intg + step_size_to_intg
        np.clip(state_to_intg, a_min=0, a_max=2 ** 23 - 1, out=state_to_intg)

        # WTA spike indices when threshold is exceeded
        wta_spk_idx = np.where(state_to_intg >= self.theta)  # Exceeds threshold
        # Spiking neuron voltages go in refractory (if neg_tau_ref < 0)
        state_to_intg[wta_spk_idx] = self.neg_tau_ref  # Post spk refractory
        spk_hist_to_intg[wta_spk_idx] |= 1

        # Assign all temporary states to state Vars
        self.state[intg_idx] = state_to_intg
        self.cnstr_intg[intg_idx] = cnstr_to_intg
        self.spk_hist[intg_idx] = spk_hist_to_intg

        return wta_spk_idx

    def _refractory_dynamics(self, rfct_idx):

        # Split/fork state variables u, v, beta
        state_in_rfct = self.state[rfct_idx]  # voltages in refractory
        cnstr_in_rfct = self.cnstr_intg[rfct_idx]  # currents in refractory
        spk_hist_in_rfct = self.spk_hist[rfct_idx]  # beta in refractory

        # Refractory dynamics
        state_in_rfct += 1  # voltage increments by 1 every step
        spk_hist_in_rfct |= 3
        # Second set of unsatisfied WTA indices based on v and u in refractory
        wta_unsat_idx_2 = \
            np.where(np.logical_or(state_in_rfct == 0, cnstr_in_rfct < 0))

        # Reset voltage of unsatisfied WTA in refractory
        state_in_rfct[wta_unsat_idx_2] = 0
        spk_hist_in_rfct[wta_unsat_idx_2] &= 2

        # Assign all temporary states to state Vars
        self.state[rfct_idx] = state_in_rfct
        self.cnstr_intg[rfct_idx] = cnstr_in_rfct
        self.spk_hist[rfct_idx] = spk_hist_in_rfct

        return wta_unsat_idx_2

    @abstractmethod
    def _gen_sig_spks(self, spk_hist_buffer):
        raise NotImplementedError("Abstract method not implemented for "
                                  "abstract class.")

    def _gen_wta_spks(self, spk_hist_buffer):
        # Indices of WTA neurons signifying unsatisfied constraints, based on
        # buffered history from previous timestep
        wta_unsat_prev_ts_idx = np.where(np.logical_and(spk_hist_buffer == 1,
                                                        self.cnstr_intg < 0))

        # Reset voltages of unsatisfied WTA
        self.state[wta_unsat_prev_ts_idx] = 0
        # indices of neurons to be integrated:
        intg_idx = np.where(self.state >= 0)
        # indices of neurons in refractory:
        rfct_idx = np.where(self.state < 0)

        # Indices of WTA neurons that will spike and enter refractory
        wta_spk_idx = self._integration_dynamics(intg_idx)

        # Indices of WTA neurons coming out of refractory or those signifying
        # unsatisfied constraints
        wta_rfct_end_or_unsat_idx = self._refractory_dynamics(rfct_idx) if \
            self.neg_tau_ref != 0 else (np.array([], dtype=np.int32),)

        s_wta = np.zeros_like(self.state)
        s_wta[wta_spk_idx] = 1
        s_wta[wta_unsat_prev_ts_idx] = -1
        s_wta[wta_rfct_end_or_unsat_idx] = -1

        return s_wta

    def run_spk(self) -> None:
        # Receive synaptic input
        self.a_in_data = self.a_in.recv()

        # Add the incoming activation and saturate to min-max limits
        np.clip(self.cnstr_intg + self.a_in_data, a_min=-2 ** 23,
                a_max=2 ** 23 - 1, out=self.cnstr_intg)

        # !! Side effect: Changes self.beta !!
        spk_hist_buffer = self._update_buffers()

        # Generate Sigma spikes
        s_sig = self._gen_sig_spks(spk_hist_buffer)

        # Generate WTA spikes
        s_wta = self._gen_wta_spks(spk_hist_buffer)

        # Send out spikes
        self.s_sig_out.send(s_sig)
        self.s_wta_out.send(s_wta)


@implements(proc=CspScif, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyModelCspScifFixed(AbstractPyModelScifFixed):
    """Fixed point implementation of Stochastic Constraint Integrate and
    Fire (SCIF) neuron for solving CSP problems.
    """

    def _gen_sig_spks(self, spk_hist_buffer):
        s_sig = np.zeros_like(self.state)
        # Gather spike and unsatisfied indices for summation axons
        sig_unsat_idx = np.where(spk_hist_buffer == 2)
        sig_spk_idx = np.where(np.logical_and(spk_hist_buffer == 1,
                                              self.cnstr_intg == 0))

        # Assign sigma spikes (+/- 1)
        s_sig[sig_unsat_idx] = -1
        s_sig[sig_spk_idx] = 1

        return s_sig


@implements(proc=QuboScif, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyModelQuboScifFixed(AbstractPyModelScifFixed):
    """Fixed point implementation of Stochastic Constraint Integrate and
        Fire (SCIF) neuron for solving QUBO problems.
    """

    cost_diagonal: np.ndarray = LavaPyType(np.ndarray, int, precision=24)

    def _gen_sig_spks(self, spk_hist_buffer):
        s_sig = np.zeros_like(self.state)
        # If we have fired in the previous time-step, we send out the local
        # cost now, i.e., when spk_hist_buffer == 1
        sig_spk_idx = np.where(spk_hist_buffer == 1)
        # Compute the local cost
        s_sig[sig_spk_idx] = self.cost_diagonal[sig_spk_idx] + \
            self.a_in_data[sig_spk_idx]

        return s_sig
