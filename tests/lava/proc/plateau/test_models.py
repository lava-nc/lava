# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/


import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.proc.plateau.process import Plateau
from lava.proc.dense.process import Dense
from lava.magma.core.run_configs import Loihi2SimCfg, RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.tests.lava.proc.lif.test_models import VecSendProcess, VecRecvProcess


class SpikeGen(AbstractProcess):
    """Process for sending spikes at user-supplied time steps.

    Parameters
    ----------
    spikes_in: list[list], list of lists containing spike times
    runtime: int, number of timesteps for the generator to store spikes
    """
    def __init__(self, spikes_in, runtime):
        super().__init__()
        n = len(spikes_in)
        self.shape = (n,)
        spike_data = np.zeros(shape=(n, runtime))
        for i in range(n):
            for t in range(1, runtime + 1):
                if t in spikes_in[i]:
                    spike_data[i, t - 1] = 1
        self.s_out = OutPort(shape=self.shape)
        self.spike_data = Var(shape=(n, runtime), init=spike_data)


@implements(proc=SpikeGen, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PySpikeGenModel(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    spike_data: np.ndarray = LavaPyType(np.ndarray, float)

    def run_spk(self):
        """Send the appropriate spikes for the given time step
        """
        self.s_out.send(self.spike_data[:, self.time_step - 1])


class TestPlateauProcessModelsFixed(unittest.TestCase):
    """Tests for the fixed point Plateau process models."""
    def test_fixed_max_decay(self):
        """
        Tests fixed point Plateau with max voltage decays.
        """
        shape = (3,)
        num_steps = 20
        spikes_in_dend = [
            [5],
            [5],
            [5],
        ]
        spikes_in_soma = [
            [3],
            [10],
            [17]
        ]
        sg_dend = SpikeGen(spikes_in=spikes_in_dend, runtime=num_steps)
        sg_soma = SpikeGen(spikes_in=spikes_in_soma, runtime=num_steps)
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
        spikes_in_dend = [[3]]
        sg_dend = SpikeGen(spikes_in=spikes_in_dend, runtime=num_steps)
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
        for t in range(num_steps):
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
        spikes_in = [[1]]
        sg_dend = SpikeGen(spikes_in=spikes_in, runtime=num_steps)
        sg_soma = SpikeGen(spikes_in=spikes_in, runtime=num_steps)
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
        for t in range(num_steps):
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
