# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi1SimCfg
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi


class TestSTDPSim(unittest.TestCase):
    def test_stdp_fixed_point(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=-2,
            A_minus=4,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
            rng_seed=0,
        )

        size = 1
        weights_init = np.eye(size) * 1

        lif_0 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=25000)

        dense = Dense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(dense.s_in_bap)

        num_steps = 100

        run_cfg = Loihi1SimCfg(select_tag="fixed_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        # np.testing.assert_almost_equal(weight_after_run, np.array([[48]]))
        np.testing.assert_almost_equal(weight_after_run, np.array([[60]]))

    def test_stdp_floating_point(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=-2,
            A_minus=1,
            tau_plus=10,
            tau_minus=10,
            t_epoch=1,
        )

        size = 1
        weights_init = np.eye(size) * 0

        lif_0 = LIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.1)

        dense = Dense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.15)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(dense.s_in_bap)

        num_steps = 100

        run_cfg = Loihi1SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(
            weight_after_run, np.array([[-79.35744962]])
        )
