# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import unittest
import numpy as np
from lava.proc.plateau.process import Plateau
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer as Source
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.tests.lava.proc.lif.test_models import VecRecvProcess


def create_spike_source(spike_list, n_indices, n_timesteps):
    """Use list of spikes [(idx, timestep), ...] to create a RingBuffer source
    with data shape (n_indices, n_timesteps) and spikes at all specified points
    in the spike_list.
    """
    data = np.zeros(shape=(n_indices, n_timesteps))
    for idx, timestep in spike_list:
        data[idx, timestep - 1] = 1
    return Source(data=data)


class TestPlateauProcessModelsFixed(unittest.TestCase):
    """Tests for the fixed point Plateau process models."""
    def test_fixed_max_decay(self):
        """
        Tests fixed point Plateau with max voltage decays.
        """
        shape = (3,)
        num_steps = 20
        spikes_in_dend = [(0, 5), (1, 5), (2, 5)]
        spikes_in_soma = [(0, 3), (1, 10), (2, 17)]
        sg_dend = create_spike_source(spikes_in_dend, shape[0], num_steps)
        sg_soma = create_spike_source(spikes_in_soma, shape[0], num_steps)
        dense_dend = Dense(weights=2 * np.diag(np.ones(shape=shape)))
        dense_soma = Dense(weights=2 * np.diag(np.ones(shape=shape)))
        plat = Plateau(
            shape=shape,
            dv_dend=4095,
            dv_soma=4095,
            vth_soma=1,
            vth_dend=1,
            up_dur=10
        )
        vr = VecRecvProcess(shape=(num_steps, shape[0]))
        sg_dend.s_out.connect(dense_dend.s_in)
        sg_soma.s_out.connect(dense_soma.s_in)
        dense_dend.a_out.connect(plat.a_dend_in)
        dense_soma.a_out.connect(plat.a_soma_in)
        plat.s_out.connect(vr.s_in)
        # run model
        plat.run(RunSteps(num_steps), Loihi2SimCfg(select_tag='fixed_pt'))
        test_spk_data = vr.spk_data.get()
        plat.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        # Neuron 2 should spike when receiving soma input
        expected_spk_data[10, 1] = 1
        self.assertTrue(np.all(expected_spk_data == test_spk_data))

    def test_up_dur(self):
        """
        Tests that the UP state lasts for the time specified by the model.
        Checks that up_state decreases by one each time step after activation.
        """
        shape = (1,)
        num_steps = 10
        spikes_in_dend = [(0, 3)]
        sg_dend = create_spike_source(spikes_in_dend, shape[0], num_steps)
        dense_dend = Dense(weights=2 * (np.diag(np.ones(shape=shape))))
        plat = Plateau(
            shape=shape,
            dv_dend=4095,
            dv_soma=4095,
            vth_soma=1,
            vth_dend=1,
            up_dur=5
        )
        sg_dend.s_out.connect(dense_dend.s_in)
        dense_dend.a_out.connect(plat.a_dend_in)
        # run model
        test_up_state = []
        for _ in range(num_steps):
            plat.run(RunSteps(1), Loihi2SimCfg(select_tag='fixed_pt'))
            test_up_state.append(plat.up_state.get().astype(int)[0])
        plat.stop()
        # Gold standard for the test
        # UP state active time steps 4 - 9 (5 timesteps)
        # this is delayed by one b.c. of the Dense process
        expected_up_state = [0, 0, 0, 5, 4, 3, 2, 1, 0, 0]
        self.assertListEqual(expected_up_state, test_up_state)

    def test_fixed_dvs(self):
        """
        Tests fixed point Plateau voltage decays.
        """
        shape = (1,)
        num_steps = 10
        spikes_in = [(0, 1)]
        sg_dend = create_spike_source(spikes_in, shape[0], num_steps)
        sg_soma = create_spike_source(spikes_in, shape[0], num_steps)
        dense_dend = Dense(weights=100 * np.diag(np.ones(shape=shape)))
        dense_soma = Dense(weights=100 * np.diag(np.ones(shape=shape)))
        plat = Plateau(
            shape=shape,
            dv_dend=2048,
            dv_soma=1024,
            vth_soma=100,
            vth_dend=100,
            up_dur=10
        )
        sg_dend.s_out.connect(dense_dend.s_in)
        sg_soma.s_out.connect(dense_soma.s_in)
        dense_dend.a_out.connect(plat.a_dend_in)
        dense_soma.a_out.connect(plat.a_soma_in)
        # run model
        test_v_dend = []
        test_v_soma = []
        for _ in range(num_steps):
            plat.run(RunSteps(1), Loihi2SimCfg(select_tag='fixed_pt'))
            test_v_dend.append(plat.v_dend.get().astype(int)[0])
            test_v_soma.append(plat.v_soma.get().astype(int)[0])
        plat.stop()
        # Gold standard for the test
        # 100<<6 = 6400 -- initial value at time step 2
        expected_v_dend = [
            0, 6400, 3198, 1598, 798, 398, 198, 98, 48, 23
        ]
        expected_v_soma = [
            0, 6400, 4798, 3597, 2696, 2021, 1515, 1135, 850, 637
        ]
        self.assertListEqual(expected_v_dend, test_v_dend)
        self.assertListEqual(expected_v_soma, test_v_soma)
