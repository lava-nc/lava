# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import Loihi2SimCfg, RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc import io
from lava.proc.filter.process import ExpFilter


class TestLIFProcessModelsFloat(unittest.TestCase):
    """Tests for floating point ProcessModels of LIF"""
    def test_float_pm_no_decay(self):
        """
        Tests floating point LIF ProcessModel with no current or voltage
        decay and neurons driven by internal biases.
        """
        shape = (10,)
        num_steps = 10
        input = np.random.rand(num_steps, *shape)
        input[input < 0.9] = 0
        input = np.ceil(input).astype(int)

        filter = ExpFilter(shape=(100,),
                   tau=0.5,
                   name="ExpFilter")



        # Set up external input to 0
        sps = VecSendProcess(shape=shape, num_steps=num_steps,
                             vec_to_send=np.zeros(shape, dtype=float),
                             send_at_times=np.ones((num_steps,), dtype=bool))
        # Set up bias = 1 * 2**1 = 2. and threshold = 4.
        # du and dv = 0 => bias driven neurons spike at every 2nd time-step.
        lif = LIF(shape=shape,
                  du=0.,
                  dv=0.,
                  bias_mant=np.ones(shape, dtype=float),
                  bias_exp=np.ones(shape, dtype=float),
                  vth=4.)
        # Receive neuron spikes
        spr = VecRecvProcess(shape=(num_steps, shape[0]))
        sps.s_out.connect(lif.a_in)
        lif.s_out.connect(spr.s_in)
        # Configure execution and run
        rcnd = RunSteps(num_steps=num_steps)
        rcfg = LifRunConfig(select_tag='floating_pt')
        lif.run(condition=rcnd, run_cfg=rcfg)
        # Gather spike data and stop
        spk_data_through_run = spr.spk_data.get()
        lif.stop()
        # Gold standard for the test
        expected_spk_data = np.zeros((num_steps, shape[0]))
        expected_spk_data[4:10:5, :] = 1.
        self.assertTrue(np.all(expected_spk_data == spk_data_through_run))