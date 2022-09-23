
# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2022 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
import sys
import unittest
import numpy as np
from typing import Tuple, Dict

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.scif.process import CspScif, QuboScif
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer as SpikeSource

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False


class TestCspScifModels(unittest.TestCase):
    """Tests for CspScif neuron"""

    def run_test(
        self,
        num_steps: int,
        num_neurons: int,
        step_size: int,
        theta: int,
        neg_tau_ref: int,
        wt: int,
        t_inj_spk: Dict[int, int],  # time_step -> payload dict to inject
        tag: str = 'fixed_pt'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        spk_src = SpikeSource(data=np.array([[0] * num_neurons]).reshape(
            num_neurons, 1).astype(int))
        dense_in = Dense(weights=(-1) * np.eye(num_neurons),
                         num_message_bits=16)
        csp_scif = CspScif(shape=(num_neurons,),
                           step_size=step_size,
                           theta=theta,
                           neg_tau_ref=neg_tau_ref)
        dense_wta = Dense(weights=wt * np.eye(num_neurons),
                          num_message_bits=16)
        dense_sig = Dense(weights=wt * np.eye(num_neurons),
                          num_message_bits=16)
        lif_wta = LIF(shape=(num_neurons,),
                      du=4095,
                      dv=4096,
                      bias_mant=0,
                      vth=2 ** 17 - 1)
        lif_sig = LIF(shape=(num_neurons,),
                      du=4095,
                      dv=4096,
                      bias_mant=0,
                      vth=2 ** 17 - 1)
        spk_src.s_out.connect(dense_in.s_in)
        dense_in.a_out.connect(csp_scif.a_in)
        csp_scif.s_wta_out.connect(dense_wta.s_in)
        csp_scif.s_sig_out.connect(dense_sig.s_in)
        dense_wta.a_out.connect(lif_wta.a_in)
        dense_sig.a_out.connect(lif_sig.a_in)

        run_condition = RunSteps(num_steps=1)
        run_config = Loihi2SimCfg(select_tag=tag)

        volts_scif = []
        volts_lif_wta = []
        volts_lif_sig = []
        for j in range(num_steps):
            if j + 1 in t_inj_spk:
                spk_src.data.set(np.array(
                    [[t_inj_spk[j + 1]] * num_neurons]).astype(int))
            csp_scif.run(condition=run_condition, run_cfg=run_config)
            spk_src.data.set(np.array([[0] * num_neurons]).astype(int))
            volts_scif.append(csp_scif.state.get())
            # Get the voltage of LIF attached to WTA
            v_wta = lif_wta.v.get()
            # Transform the voltage into +/- 1 spike
            v_wta = (v_wta / wt).astype(int)  # De-scale the weight
            v_wta = np.right_shift(v_wta, 6)  # downshift DendAccum's effect
            # Append to list
            volts_lif_wta.append(v_wta)
            # Get the voltage of LIF attached to Sig
            v_sig = lif_sig.v.get()
            # Transform the voltage into +/- 1 spike
            v_sig = (v_sig / wt).astype(int)  # De-scale the weight
            v_sig = np.right_shift(v_sig, 6)  # downshift DendAccum's effect
            # Append to list
            volts_lif_sig.append(v_sig)

        csp_scif.stop()

        return np.array(volts_scif).astype(int), \
            np.array(volts_lif_wta).astype(int), \
            np.array(volts_lif_sig).astype(int)

    def test_scif_fixed_pt_no_noise(self) -> None:
        """Test a single SCIF neuron without noise, but with a constant bias.
        The neuron is expected to spike with a regular period, on WTA as well as
        Sigma axons. After excitatory spikes on two consecutive time-steps, the
        neuron goes into inhibition and sends 2 inhibitory spikes of payload -1
        at the end of its refractory period.
        """
        num_neurons = np.random.randint(1, 11)
        step_size = 1
        theta = 4
        neg_tau_ref = -5
        wt = 2
        total_period = theta // step_size - neg_tau_ref
        num_epochs = 10
        num_steps = num_epochs * total_period + (theta // step_size)
        v_scif, v_lif_wta, v_lif_sig = self.run_test(num_steps=num_steps,
                                                     num_neurons=num_neurons,
                                                     step_size=step_size,
                                                     theta=theta,
                                                     neg_tau_ref=neg_tau_ref,
                                                     wt=wt,
                                                     t_inj_spk={})
        spk_idxs = np.array([theta // step_size - 1 + j * total_period for j in
                             range(num_epochs)]).astype(int)
        wta_pos_spk_idxs = spk_idxs + 1
        sig_pos_spk_idxs = wta_pos_spk_idxs + 1
        wta_neg_spk_idxs = wta_pos_spk_idxs - neg_tau_ref
        sig_neg_spk_idxs = sig_pos_spk_idxs - neg_tau_ref
        self.assertTrue(np.all(v_scif[spk_idxs] == neg_tau_ref))
        self.assertTrue(np.all(v_lif_wta[wta_pos_spk_idxs] == 1))
        self.assertTrue(np.all(v_lif_sig[sig_pos_spk_idxs] == 1))
        self.assertTrue(np.all(v_lif_wta[wta_neg_spk_idxs] == -1))
        self.assertTrue(np.all(v_lif_sig[sig_neg_spk_idxs] == -1))

    def test_scif_fp_no_noise_interrupt_rfct_mid(self) -> None:
        """
        Test a single SCIF neuron without LFSR noise, but with a constant bias.

        An inhibitory spike is injected in the middle of the refractory
        period after the neuron spikes for the first time. The inhibition
        interrupts the refractory period. The neuron issues negative spikes
        at WTA and Sigma axons on consecutive time-steps.

        An excitatory spike is injected to nullify the inhibition and neuron
        starts spiking periodically again.
        """
        num_neurons = np.random.randint(1, 11)
        step_size = 1
        theta = 4
        neg_tau_ref = -5
        wt = 2
        t_inj_spk = {7: 1, 11: -1}
        inj_times = list(t_inj_spk.keys())
        total_period = (theta // step_size) - neg_tau_ref
        num_epochs = 5
        num_steps = \
            (theta // step_size) + num_epochs * total_period + inj_times[1]
        v_scif, v_lif_wta, v_lif_sig = self.run_test(num_steps=num_steps,
                                                     num_neurons=num_neurons,
                                                     step_size=step_size,
                                                     theta=theta,
                                                     neg_tau_ref=neg_tau_ref,
                                                     wt=wt,
                                                     t_inj_spk=t_inj_spk)
        # Test pre-inhibitory-injection SCIF voltage and spiking
        spk_idxs_pre_inj = np.array([theta // step_size]).astype(int) - 1
        wta_pos_spk_pre_inj = spk_idxs_pre_inj + 1
        sig_pos_spk_pre_inj = wta_pos_spk_pre_inj + 1
        inh_inj = inj_times[0]
        wta_neg_spk_rfct_interrupt = inh_inj + 1
        sig_neg_spk_rfct_interrupt = wta_neg_spk_rfct_interrupt + 1
        self.assertTrue(np.all(v_scif[spk_idxs_pre_inj] == neg_tau_ref))
        self.assertTrue(np.all(v_lif_wta[wta_pos_spk_pre_inj] == 1))
        self.assertTrue(np.all(v_lif_sig[sig_pos_spk_pre_inj] == 1))
        self.assertTrue(np.all(v_scif[inh_inj] == 0))
        self.assertTrue(np.all(v_lif_wta[wta_neg_spk_rfct_interrupt] == -1))
        self.assertTrue(np.all(v_lif_sig[sig_neg_spk_rfct_interrupt] == -1))
        # Test post-inhibitory-injection SCIF voltage and spiking
        idx_lst = [inj_times[1] + (theta // step_size) - 1 + j * total_period
                   for j in range(num_epochs)]
        spk_idxs_post_inj = np.array(idx_lst).astype(int)
        wta_pos_spk_idxs = spk_idxs_post_inj + 1
        sig_pos_spk_idxs = wta_pos_spk_idxs + 1
        wta_neg_spk_idxs = wta_pos_spk_idxs - neg_tau_ref
        sig_neg_spk_idxs = sig_pos_spk_idxs - neg_tau_ref
        self.assertTrue(np.all(v_scif[spk_idxs_post_inj] == neg_tau_ref))
        self.assertTrue(np.all(v_lif_wta[wta_pos_spk_idxs] == 1))
        self.assertTrue(np.all(v_lif_sig[sig_pos_spk_idxs] == 1))
        self.assertTrue(np.all(v_lif_wta[wta_neg_spk_idxs] == -1))
        self.assertTrue(np.all(v_lif_sig[sig_neg_spk_idxs] == -1))

    def test_scif_fp_no_noise_interrupt_rfct_beg(self) -> None:
        """
        Test a single SCIF neuron without LFSR noise, but with a constant bias.

        An inhibitory spike is injected at the very start of the refractory
        period after the neuron spikes for the first time. The inhibition
        interrupts the refractory period. The neuron issues a negative spike
        at WTA axons to nullify its positive spike. No spikes are issued on
        the Sigma axon.

        An excitatory spike is injected to nullify the inhibition and neuron
        starts spiking periodically again.
        """
        num_neurons = np.random.randint(1, 11)
        step_size = 1
        theta = 4
        neg_tau_ref = -5
        wt = 2
        t_inj_spk = {4: 1, 8: -1}
        inj_times = list(t_inj_spk.keys())
        total_period = theta // step_size - neg_tau_ref
        num_epochs = 5
        num_steps = \
            num_epochs * total_period + (theta // step_size) + inj_times[1]
        v_scif, v_lif_wta, v_lif_sig = self.run_test(num_steps=num_steps,
                                                     num_neurons=num_neurons,
                                                     step_size=step_size,
                                                     theta=theta,
                                                     neg_tau_ref=neg_tau_ref,
                                                     wt=wt,
                                                     t_inj_spk=t_inj_spk)
        # Test pre-inhibitory-injection SCIF voltage and spiking
        spk_idxs_pre_inj = np.array([theta // step_size]).astype(int) - 1
        wta_pos_spk_pre_inj = spk_idxs_pre_inj + 1
        wta_neg_spk_pre_inj = wta_pos_spk_pre_inj + 1
        inh_inj = inj_times[0]
        self.assertTrue(np.all(v_scif[spk_idxs_pre_inj] == neg_tau_ref))
        self.assertTrue(np.all(v_lif_wta[wta_pos_spk_pre_inj] == 1))
        self.assertTrue(np.all(v_lif_wta[wta_neg_spk_pre_inj] == -1))
        self.assertTrue(np.all(v_lif_sig[wta_neg_spk_pre_inj] == 0))
        self.assertTrue(np.all(v_scif[inh_inj] == 0))

        # Test post-inhibitory-injection SCIF voltage and spiking
        idx_lst = [inj_times[1] + (theta // step_size) - 1 + j * total_period
                   for j in range(num_epochs)]
        spk_idxs_post_inj = np.array(idx_lst).astype(int)
        wta_pos_spk_idxs = spk_idxs_post_inj + 1
        sig_pos_spk_idxs = wta_pos_spk_idxs + 1
        wta_neg_spk_idxs = wta_pos_spk_idxs - neg_tau_ref
        sig_neg_spk_idxs = sig_pos_spk_idxs - neg_tau_ref
        self.assertTrue(np.all(v_scif[spk_idxs_post_inj] == neg_tau_ref))
        self.assertTrue(np.all(v_lif_wta[wta_pos_spk_idxs] == 1))
        self.assertTrue(np.all(v_lif_sig[sig_pos_spk_idxs] == 1))
        self.assertTrue(np.all(v_lif_wta[wta_neg_spk_idxs] == -1))
        self.assertTrue(np.all(v_lif_sig[sig_neg_spk_idxs] == -1))


class TestQuboScifModels(unittest.TestCase):
    """Tests for sigma delta neuron"""

    def run_test(
        self,
        num_steps: int,
        num_neurons: int,
        cost_diag: np.ndarray,
        step_size: int,
        theta: int,
        neg_tau_ref: int,
        wt: int,
        t_inj_spk: Dict[int, int],  # time_step -> payload dict to inject
        tag: str = 'fixed_pt'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        spk_src = SpikeSource(data=np.array([[0] * num_neurons]).reshape(
            num_neurons, 1).astype(int))
        dense_in = Dense(weights=(-1) * np.eye(num_neurons),
                         num_message_bits=16)
        qubo_scif = QuboScif(shape=(num_neurons,),
                             cost_diag=cost_diag,
                             step_size=step_size,
                             theta=theta,
                             neg_tau_ref=neg_tau_ref)
        dense_wta = Dense(weights=wt * np.eye(num_neurons),
                          num_message_bits=16)
        dense_sig = Dense(weights=wt * np.eye(num_neurons),
                          num_message_bits=16)
        lif_wta = LIF(shape=(num_neurons,),
                      du=4095,
                      dv=4096,
                      bias_mant=0,
                      vth=2 ** 17 - 1)
        lif_sig = LIF(shape=(num_neurons,),
                      du=4095,
                      dv=4096,
                      bias_mant=0,
                      vth=2 ** 17 - 1)
        spk_src.s_out.connect(dense_in.s_in)
        dense_in.a_out.connect(qubo_scif.a_in)
        qubo_scif.s_wta_out.connect(dense_wta.s_in)
        qubo_scif.s_sig_out.connect(dense_sig.s_in)
        dense_wta.a_out.connect(lif_wta.a_in)
        dense_sig.a_out.connect(lif_sig.a_in)

        run_condition = RunSteps(num_steps=1)
        run_config = Loihi2SimCfg(select_tag=tag)

        volts_scif = []
        volts_lif_wta = []
        volts_lif_sig = []
        for j in range(num_steps):
            if j + 1 in t_inj_spk:
                spk_src.data.set(
                    np.array([[t_inj_spk[j + 1]] * num_neurons]).astype(int))
            qubo_scif.run(condition=run_condition, run_cfg=run_config)
            spk_src.data.set(np.array([[0] * num_neurons]).astype(int))
            volts_scif.append(qubo_scif.state.get())
            # Get the voltage of LIF attached to WTA
            v_wta = lif_wta.v.get()
            # Transform the voltage into +/- 1 spike
            v_wta = (v_wta / wt).astype(int)  # De-scale the weight
            v_wta = np.right_shift(v_wta, 6)  # downshift DendAccum's effect
            # Append to list
            volts_lif_wta.append(v_wta)
            # Get the voltage of LIF attached to Sig
            v_sig = lif_sig.v.get()
            # Transform the voltage into +/- 1 spike
            v_sig = (v_sig / wt).astype(int)  # De-scale the weight
            v_sig = np.right_shift(v_sig, 6)  # downshift DendAccum's effect
            # Append to list
            volts_lif_sig.append(v_sig)

        qubo_scif.stop()

        return np.array(volts_scif).astype(int), \
            np.array(volts_lif_wta).astype(int), \
            np.array(volts_lif_sig).astype(int)

    def test_scif_fixed_pt_no_noise(self) -> None:
        """Test a single SCIF neuron without noise, but with a constant bias.
        The neuron is expected to spike with a regular period, on WTA as well as
        Sigma axons. After excitatory spikes on two consecutive time-steps, the
        neuron goes into inhibition and sends 2 inhibitory spikes of payload -1
        at the end of its refractory period.
        """
        num_neurons = np.random.randint(1, 11)
        cost_diag = np.arange(1, num_neurons + 1)
        step_size = 1
        theta = 4
        neg_tau_ref = 0
        wt = 2
        total_period = theta // step_size - neg_tau_ref
        num_epochs = 10
        num_steps = num_epochs * total_period + (theta // step_size)
        v_scif, v_lif_wta, v_lif_sig = self.run_test(num_steps=num_steps,
                                                     num_neurons=num_neurons,
                                                     cost_diag=cost_diag,
                                                     step_size=step_size,
                                                     theta=theta,
                                                     neg_tau_ref=neg_tau_ref,
                                                     wt=wt,
                                                     t_inj_spk={})
        spk_idxs = np.array([theta // step_size - 1 + j * total_period for j in
                             range(num_epochs)]).astype(int)
        wta_pos_spk_idxs = spk_idxs + 1
        sig_pos_spk_idxs = wta_pos_spk_idxs + 1
        self.assertTrue(np.all(v_scif[spk_idxs] == neg_tau_ref))
        self.assertTrue(np.all(v_lif_wta[wta_pos_spk_idxs] == 1))
        self.assertTrue(np.all(v_lif_sig[sig_pos_spk_idxs] == cost_diag))

    def test_scif_fp_no_noise_interrupt_rfct_mid(self) -> None:
        """
        Test a single SCIF neuron without LFSR noise, but with a constant bias.

        An inhibitory spike is injected in the middle of the refractory
        period after the neuron spikes for the first time. The inhibition
        interrupts the refractory period. The neuron issues negative spikes
        at WTA and Sigma axons on consecutive time-steps.

        An excitatory spike is injected to nullify the inhibition and neuron
        starts spiking periodically again.
        """
        num_neurons = np.random.randint(1, 11)
        cost_diag = np.arange(1, num_neurons + 1)
        step_size = 1
        theta = 4
        neg_tau_ref = 0
        wt = 2
        t_inj_spk = {4: -1, 6: -1, 8: 2}
        inj_times = list(t_inj_spk.keys())
        total_period = (theta // step_size) - neg_tau_ref
        num_epochs = 5
        num_steps = \
            (theta // step_size) + num_epochs * total_period + inj_times[1]
        v_scif, v_lif_wta, v_lif_sig = self.run_test(num_steps=num_steps,
                                                     num_neurons=num_neurons,
                                                     cost_diag=cost_diag,
                                                     step_size=step_size,
                                                     theta=theta,
                                                     neg_tau_ref=neg_tau_ref,
                                                     wt=wt,
                                                     t_inj_spk=t_inj_spk)
        # Test pre-inhibitory-injection SCIF voltage and spiking
        spk_idxs_pre_inj = np.array([theta // step_size]).astype(int) - 1
        wta_pos_spk_pre_inj = spk_idxs_pre_inj + 1
        sig_pos_spk_pre_inj = wta_pos_spk_pre_inj + 1
        inh_inj = inj_times[0]
        wta_spk_rfct_interrupt = inh_inj
        sig_spk_rfct_interrupt = wta_spk_rfct_interrupt + 1
        self.assertTrue(np.all(v_scif[spk_idxs_pre_inj] == neg_tau_ref))
        self.assertTrue(np.all(v_lif_wta[wta_pos_spk_pre_inj] == 1))
        self.assertTrue(np.all(
            v_lif_sig[sig_pos_spk_pre_inj] == cost_diag + wt * step_size))
        v_gt_inh_inj = (inh_inj - spk_idxs_pre_inj + 1) - t_inj_spk[inh_inj]
        self.assertTrue(np.all(v_scif[inh_inj] == v_gt_inh_inj))
        self.assertTrue(np.all(v_lif_wta[wta_spk_rfct_interrupt] == 1))
        self.assertTrue(np.all(
            v_lif_sig[sig_spk_rfct_interrupt] == cost_diag + wt * step_size))
        # Test post-inhibitory-injection SCIF voltage and spiking
        idx_lst = [inj_times[2] + (theta // step_size) - 1 + j * total_period
                   for j in range(num_epochs)]
        spk_idxs_post_inj = np.array(idx_lst).astype(int)
        wta_pos_spk_idxs = spk_idxs_post_inj + 1
        sig_pos_spk_idxs = wta_pos_spk_idxs + 1
        self.assertTrue(np.all(v_scif[spk_idxs_post_inj] == neg_tau_ref))
        self.assertTrue(np.all(v_lif_wta[wta_pos_spk_idxs] == 1))
        self.assertTrue(np.all(
            v_lif_sig[sig_pos_spk_idxs] == cost_diag))
