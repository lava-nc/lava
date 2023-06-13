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


def rf_dynamics(real_state, imag_state, sin_decay, cos_decay, real_input,
                imag_input, decay_bits):
    """Fixed pt rf dynamics test function

    Parameters
    ----------
    real_state : float
        Starting real voltage
    imag_state : float
        Starting imag voltage
    sin_decay : float
        (1 - alpha) * sin(theta)
    cos_decay : float
        (1 - alpha) * cos(theta)
    real_input : np.array
        real input into neuron
    imag_input : np.array
        imag input into neuron
    decay_bits : int
        Downscale factor applied after voltage decay

    Returns
    -------
    np.array, np.array
        Observed real and imaginary voltage
    """
    scale_fac = (1 << decay_bits)

    def rtz(x):  # downscale voltage
        return (int(x / scale_fac) if x > 0
                else -int((-x) / scale_fac))
    real = np.zeros_like(real_input)
    imag = np.zeros_like(imag_input)
    for n in range(len(real_input)):
        real[n] = rtz(cos_decay * real_state) \
            - rtz(sin_decay * imag_state) \
            + real_input[n]

        imag[n] = rtz(sin_decay * real_state) \
            + rtz(cos_decay * imag_state) \
            + imag_input[n]
        real_state = real[n]
        imag_state = imag[n]

    return real, imag


class TestrfProcessModels(unittest.TestCase):
    """Tests for RF Process Model"""

    def run_test(
        self,
        period: float,
        alpha: float,
        input_: np.ndarray,
        state_exp: int = 0,
        decay_bits: int = 0,
        vth: float = 1,
        tag: str = 'floating_pt',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        input_ = np.int32(input_.reshape(1, -1))
        num_steps = input_.size
        source = io.source.RingBuffer(data=input_)
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

        sink.run(condition=run_condition, run_cfg=run_config)

        s_out = sink.data.get()
        real = real_monitor.get_data()[rf.name]["real"]
        imag = imag_monitor.get_data()[rf.name]["imag"]
        rf.stop()

        return input_, real, imag, s_out

    def test_float_no_decay(self):
        """Verify that a neuron with no voltage decay spikes
            periodicaly due a single input pulse
        """
        period = 10
        alpha = 0

        num_steps = 100
        input_ = np.zeros(num_steps)
        input_[0] = 1.1  # spike at first timestep
        _, _, _, s_out = self.run_test(period, alpha, input_)

        # observe differences in spike times
        spike_idx = np.argwhere(s_out[0, :])
        s_diffs = [j - i for i, j in zip(spike_idx[:-1], spike_idx[1:])]

        # Ensure correct spike count
        self.assertTrue((num_steps - 1) / period, s_out.size)
        # We should see observe periodic spiking
        self.assertTrue(np.all(np.array(s_diffs) == period))

    def test_float_decay(self):
        """Neuron recieves an input pulse. The decay of the observed real
           voltage should match the alpha parameter at the real peaks.
        """
        period = 10
        alpha = .07
        vth = 1.1

        num_steps = 100
        input_ = np.zeros(num_steps)
        input_[0] = 1  # spike at first timestep
        _, real, _, _ = self.run_test(period, alpha, input_, vth=vth)

        ideal_real = np.round((1 - alpha)**np.arange(num_steps), 6)
        round_real = np.round(real.flatten(), 6)
        self.assertListEqual(round_real.tolist()[0:num_steps:period],
                             ideal_real.tolist()[0:num_steps:period])

    def test_fixed_pm_no_decay(self):
        """Test that a fixed point rf neuron with no decay matches a for-loop
           implementation of the same rf neuron. The neuron does not spike
           over the the whole period due to fixed point precision errors
        """
        alpha = 0
        vth = .5  # choose a low voltage threshold to achieve repeat spiking
        decay_bits = 12
        state_exp = 6
        period = 10

        num_steps = 100
        input_ = np.zeros(num_steps)
        input_[0] = 1  # spike at first timestep

        _, _, _, s_out = self.run_test(period, alpha, input_, vth=vth,
                                       state_exp=state_exp,
                                       decay_bits=decay_bits,
                                       tag="fixed_pt")

        # real, imag voltages
        sin_decay = (1 - alpha) * np.sin(np.pi * 2 * 1 / period)
        cos_decay = (1 - alpha) * np.cos(np.pi * 2 * 1 / period)

        # scale decays
        sin_decay = int(sin_decay * (1 << decay_bits))
        cos_decay = int(cos_decay * (1 << decay_bits))

        # Run Test RF Dynamics
        real, imag = rf_dynamics(0, 0, sin_decay, cos_decay,
                                 input_ * (1 << state_exp),
                                 np.zeros(num_steps),
                                 decay_bits)

        # spiking function needs time delayed imaginary voltage
        old_imag = np.zeros(num_steps)
        old_imag[1:num_steps] = imag[:num_steps - 1]
        expected_spikes = (real >= (vth * (1 << state_exp))) \
            * (imag >= 0) * (old_imag < 0)

        self.assertListEqual(expected_spikes.tolist(),
                             s_out.flatten().tolist())

    def test_fixed_pm_decay1(self):
        """Test that a fixed point rf neuron with decay matches
           a for-loop implementation of the same rf neuron
        """
        alpha = .05
        vth = .5  # choose a low voltage threshold to achieve repeat spiking
        decay_bits = 12
        state_exp = 6
        period = 10

        num_steps = 100
        input_ = np.zeros(num_steps)
        input_[0] = 2  # spike at first timestep

        _, _, _, s_out = self.run_test(period, alpha, input_, vth=vth,
                                       state_exp=state_exp,
                                       decay_bits=decay_bits,
                                       tag="fixed_pt")

        # real, imag voltages
        sin_decay = (1 - alpha) * np.sin(np.pi * 2 * 1 / period)
        cos_decay = (1 - alpha) * np.cos(np.pi * 2 * 1 / period)

        # scale decays
        sin_decay = int(sin_decay * (1 << decay_bits))
        cos_decay = int(cos_decay * (1 << decay_bits))

        # Run Test RF Dynamics
        real, imag = rf_dynamics(0, 0, sin_decay, cos_decay,
                                 input_ * (1 << state_exp),
                                 np.zeros(num_steps),
                                 decay_bits)

        # spiking function needs time delayed imaginary voltage
        old_imag = np.zeros(num_steps)
        old_imag[1:num_steps] = imag[:num_steps - 1]
        expected_spikes = (real >= (vth * (1 << state_exp))) \
            * (imag >= 0) * (old_imag < 0)

        self.assertListEqual(expected_spikes.tolist(),
                             s_out.flatten().tolist())

    def test_fixed_pm_decay2(self):
        """Neuron recieves an input pulse. The decay of the observed real
           voltage should match the alpha parameter. The oscillatory internal
           state of the neuron is disabled for this test by choosing a large
           neuron period
        """
        alpha = 0.07
        vth = 1.1
        decay_bits = 12
        state_exp = 6

        # Solve for a period that is large enough so that there is no
        # accumulated imaginary component. So,
        # sin_decay * (1 << state_exp) < (1 << decay_bits)
        period = np.ceil(np.pi * 2
                         / (np.arcsin(1 / ((1 << state_exp) * (1 - alpha)))))
        cos_decay = (1 - alpha) * np.cos(np.pi * 2 * 1 / period)
        cos_decay = int(cos_decay * (1 << decay_bits))

        num_steps = 100
        input_ = np.zeros(num_steps)
        input_[0] = 1  # spike at first timestep
        decay_bits = 12
        state_exp = 6

        _, real, _, _ = self.run_test(period, alpha, input_, vth=vth,
                                      state_exp=state_exp,
                                      decay_bits=decay_bits,
                                      tag="fixed_pt")

        # Repeatedly decay real voltage
        voltage = np.int32(1 * (1 << state_exp))
        ideal_real = []
        for _ in range(num_steps):
            ideal_real.append(voltage)
            voltage = np.right_shift(voltage * cos_decay, decay_bits)

        self.assertListEqual(real.flatten().tolist(), ideal_real)
