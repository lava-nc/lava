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
from lava.proc.learning_rules.three_factor_learning_rules import DopaminergicSTDPLoihi
from lava.proc.monitor.process import Monitor


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

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000, enable_learning=True)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out_bap.connect(dense.s_in_bap)

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

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.15, enable_learning=True)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out_bap.connect(dense.s_in_bap)

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

    def test_dopa_stdp_floating_point(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        def trace_callback(neuron):
            # gets called in the run_spk phase of lif_1.
            # Can be used to return the trace values of y2 and y3
            # Here the traces are dependent of the membrane potential / input current
            trace_y2 = neuron.v ** 2
            trace_y3 = np.sqrt(neuron.u)
            return trace_y2, trace_y3

        learning_rule = DopaminergicSTDPLoihi(
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

        dense = Dense(weights=weights_init, learning_rule=learning_rule, name="dense")

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.15,
                    enable_learning=True, # set to True to send out traces and bap
                    update_traces=trace_callback) # pass function to calculate traces

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out_bap.connect(dense.s_in_bap)
        lif_1.s_out_y2.connect(dense.s_in_y2)

        num_steps = 100

        weight_before_run = dense.weights.get()

        run_cfg = Loihi1SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()

        lif_0.stop()

        # np.testing.assert_almost_equal(weight_before_run, weights_init)
        # np.testing.assert_almost_equal(
        #    weight_after_run, np.array([[-79.35744962]])
        # )

    def print_dopa_stdp_floating_point(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        def trace_callback(neuron):
            trace_y2 = neuron.v
            trace_y3 = neuron.u
            return trace_y2, trace_y3

        learning_rule = DopaminergicSTDPLoihi(
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

        dense = Dense(weights=weights_init, learning_rule=learning_rule, name="dense")

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.15, enable_learning=True, update_traces=trace_callback)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out_bap.connect(dense.s_in_bap)
        lif_1.s_out_y2.connect(dense.s_in_y2)

        num_steps = 100

        mon_x = Monitor()
        mon_y = Monitor()
        mon_y2 = Monitor()
        mon_weight = Monitor()

        mon_x.probe(dense.x1, num_steps=num_steps)
        mon_y.probe(dense.y1, num_steps=num_steps)
        mon_y2.probe(dense.y2, num_steps=num_steps)
        mon_weight.probe(dense.weights, num_steps=num_steps)

        run_cfg = Loihi1SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        x1 = mon_x.get_data()['dense']['x1'][:, 0]
        y1 = mon_y.get_data()['dense']['y1'][:, 0]
        y2 = mon_y2.get_data()['dense']['y2'][:, 0]
        weight = mon_weight.get_data()['dense']['weights'][:, 0]

        weight_after_run = dense.weights.get()

        print (np.array(list(zip(x1, y1, y2, weight))))
        lif_0.stop()

        #np.testing.assert_almost_equal(weight_before_run, weights_init)
        #np.testing.assert_almost_equal(
        #    weight_after_run, np.array([[-79.35744962]])
        #)
