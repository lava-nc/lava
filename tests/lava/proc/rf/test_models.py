# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import numpy as np

from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.proc.rf.process import RF
from typing import Tuple
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.monitor.process import Monitor


class TestrfProcessModels(unittest.TestCase):
    """Tests for RF Process Model"""

    def run_test(
        self,
        period: float,
        alpha: float,
        input: np.ndarray,
        state_exp: int = 0,
        decay_bits: int = 0,
        vth: float = 1,
        tag: str = 'floating_pt',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        input = input.reshape(1, -1)
        num_steps = input.size
        source = io.source.RingBuffer(data=input)
        rf = RF(shape=(1,),
                period=period,
                alpha=alpha,
                vth=vth,
                state_exp=state_exp,
                decay_bits=decay_bits)

        real_monitor = Monitor()
        real_monitor.probe(target=rf.real, num_steps=num_steps)
        imag_monitor = Monitor()
        imag_monitor.probe(target=rf.imag, num_steps=num_steps)
        sink = io.sink.RingBuffer(shape=rf.shape, buffer=num_steps)

        source.s_out.connect(rf.a_real_in)
        rf.s_out.connect(sink.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = Loihi1SimCfg(select_tag=tag)

        rf.run(condition=run_condition, run_cfg=run_config)
    
        s_out = sink.data.get()
        real = real_monitor.get_data()[rf.name]["real"]
        imag = imag_monitor.get_data()[rf.name]["imag"]
        rf.stop()

        return input, real, imag, s_out

    def test_float_no_decay(self):
        """Neuron should spike on first input and then spike periodicaly
           for the remainder of the simulation
        """
        period = 10
        alpha = 0

        num_steps = 100
        input = np.zeros(num_steps)
        input[0] = 1  # spike at first timestep
        _, _, _, s_out = self.run_test(period, alpha, input)
        expected_spikes = np.zeros((num_steps))
        expected_spikes[10:num_steps:period] = 1
        self.assertListEqual(expected_spikes.tolist(),
                             s_out.flatten().tolist())

    def test_float_decay(self):
        """Neuron recieves an input pulse. The decay of the internal real
           voltage should match the alpha parameter.
        """
        period = 4
        alpha = .07
        vth = 1.1

        num_steps = 100
        input = np.zeros(num_steps)
        input[0] = 1  # spike at first timestep
        _, real, _, _ = self.run_test(period, alpha, input, vth=vth)
        expected_spikes = np.zeros((num_steps))
        expected_spikes[10:num_steps:period] = 1
        self.assertAlmostEquals(real.flatten()[40], (1 - alpha)**40)
