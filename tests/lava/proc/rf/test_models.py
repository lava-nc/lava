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
        """Neuron should from a single input and then spike periodicaly
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
        """Neuron recieves an input pulse and does not spike. The decay of
           the observed real voltage should match the alpha parameter.
        """
        period = 10
        alpha = .07
        vth = 1.1

        num_steps = 100
        input = np.zeros(num_steps)
        input[0] = 1  # spike at first timestep
        _, real, _, s_out = self.run_test(period, alpha, input, vth=vth)

        ideal_real = np.round((1 - alpha)**np.arange(num_steps), 4)
        round_real = np.round(real.flatten(), 4)
        self.assertListEqual(round_real.flatten().tolist()[0:num_steps:period],
                             ideal_real.tolist()[0:num_steps:period])
        self.assertListEqual(s_out.flatten().tolist(),
                             [0] * num_steps)

    def test_fixed_no_decay(self):
        """Neuron with no alpha decay should spike periodically due
           to a single input spike. We observe voltage decay caused
           by accumulated fixed point precision errors.
        """
        alpha = 0
        vth = .5
        decay_bits = 12
        state_exp = 6
        period = 10

        num_steps = 1000
        input = np.zeros(num_steps)
        input[0] = 1  # spike at first timestep
        decay_bits = 12
        state_exp = 6

        _, real, imag, s_out = self.run_test(period, alpha, input, vth=vth,
                                             state_exp=state_exp,
                                             decay_bits=decay_bits,
                                             tag="fixed_pt")

        # real, imag voltages
        ri_volt = np.zeros((2, 1), dtype=np.int32)
        ri_volt[0][0] = (1 << state_exp)  # scale initial voltage
        sin_decay = (1 - alpha) * np.sin(np.pi * 2 * 1 / period)
        cos_decay = (1 - alpha) * np.cos(np.pi * 2 * 1 / period)

        # scale decays
        sin_decay = int(sin_decay * (1 << decay_bits))
        cos_decay = int(cos_decay * (1 << decay_bits))

        # dynamics can be represented as a rotation
        rot_matrix = np.array([[cos_decay, -sin_decay],
                              [sin_decay, cos_decay]], dtype=np.int32)

        # Repeatedly rotate and simulate fixed precision ops
        ideal_voltages = []
        for _ in range(num_steps):
            ideal_voltages.append(ri_volt)
            scaled_voltage = np.matmul(rot_matrix, np.int64(ri_volt))
            ri_volt = np.sign(scaled_voltage) * np.right_shift(np.abs(
                               scaled_voltage), decay_bits)

        # Define our ground truth
        ideal_real = [i[0, 0] for i in ideal_voltages]
        ideal_imag = [i[1, 0] for i in ideal_voltages]
        expected_spikes = np.zeros((num_steps))
        expected_spikes[10:num_steps:period] = 1  # no spike at t = 0
        expected_spikes[np.array(ideal_real) < (vth * (1 << state_exp))] = 0

        self.assertListEqual(expected_spikes.tolist(),
                             s_out.flatten().tolist())
        self.assertListEqual(real.flatten().tolist(), ideal_real)
        self.assertListEqual(imag.flatten().tolist(), ideal_imag)

    def test_fixed_pm_decay(self):
        """Neuron recieves an input pulse and does not spike. The decay of
            the observed real voltage should match the alpha parameter.
            The oscillatory internal state of neuron is disabled for
            this test by choosing a large neuron period
        """
        alpha = 0.07
        vth = 1.1
        decay_bits = 12
        state_exp = 6
        # Solve for a period that is large enough so that there is no
        # accumulated imaginary component.
        # So sin_decay * (1 << state_exp) < (1 << decay_bits)
        period = np.ceil(np.pi*2/(np.arcsin(1/((1 << state_exp)*(1-alpha)))))

        num_steps = 100
        input = np.zeros(num_steps)
        input[0] = 1  # spike at first timestep
        decay_bits = 12
        state_exp = 6

        _, real, _, s_out = self.run_test(period, alpha, input, vth=vth,
                                          state_exp=state_exp,
                                          decay_bits=decay_bits,
                                          tag="fixed_pt")
        # Repeatedly decay real voltage
        voltage = np.int32(1 * (1 << state_exp))
        quantized_decay = np.int32((1 - alpha) * (1 << decay_bits))
        ideal_real = []
        for _ in range(num_steps):
            ideal_real.append(voltage)
            voltage = np.right_shift(voltage*quantized_decay, decay_bits)

        self.assertListEqual(real.flatten().tolist(), ideal_real)
        self.assertListEqual(real.flatten().tolist(), ideal_real)
        self.assertListEqual(s_out.flatten().tolist(),
                             [0] * num_steps)
