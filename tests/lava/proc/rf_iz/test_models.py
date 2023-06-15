# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.proc.rf_iz.process import RF_IZ
from typing import Tuple
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.monitor.process import Monitor


class Testrf_izProcessModels(unittest.TestCase):
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
        input_ = input_.reshape(1, -1)
        num_steps = input_.size
        source = io.source.RingBuffer(data=input_)
        rf = RF_IZ(shape=(1,),
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

        source.s_out.connect(rf.a_imag_in)
        rf.s_out.connect(sink.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = Loihi1SimCfg(select_tag=tag)

        rf.run(condition=run_condition, run_cfg=run_config)

        s_out = sink.data.get()
        real = real_monitor.get_data()[rf.name]["real"]
        imag = imag_monitor.get_data()[rf.name]["imag"]
        rf.stop()

        return input_, real, imag, s_out

    def test_float_reset(self):
        """Ensure that spikes events are followed by proper rf_iz reset
        """
        period = 10
        alpha = .07
        eps = 1e-5

        num_steps = 100
        input_ = np.zeros(num_steps)
        input_[[0, 10, 20]] = 1  # Will ensure 3 spikes
        _, real, imag, s_out = self.run_test(period, alpha, input_)
        s_out = s_out.flatten() == 1  # change to bool
        self.assertGreaterEqual(s_out.sum(), 1)  # ensure network is spiking
        self.assertListEqual(real.flatten()[s_out].tolist(),
                             [0] * np.sum(s_out))
        self.assertListEqual(imag.flatten()[s_out].tolist(),
                             [1 - eps] * np.sum(s_out))

    def test_fixed_pm_reset(self):
        """Ensure that spikes events are followed by proper rf_iz reset
           for fixed point implementation
        """
        period = 10
        alpha = .07
        eps = 1  # in fixed point 1 is the smallest value we can have
        state_exp = 6
        num_steps = 100
        input_ = np.zeros(num_steps)
        input_[[0, 10, 20]] = 1  # Will ensure 3 spikes
        _, real, imag, s_out = self.run_test(period, alpha, input_,
                                             tag="fixed_pt",
                                             state_exp=state_exp,
                                             decay_bits=12)
        s_out = s_out.flatten() == 1  # change to bool
        self.assertGreaterEqual(s_out.sum(), 1)  # ensure network is spiking
        self.assertListEqual(real.flatten()[s_out].tolist(),
                             [0] * np.sum(s_out))
        self.assertListEqual(imag.flatten()[s_out].tolist(),
                             [(1 << state_exp) - eps] * np.sum(s_out))
