# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.clp.prototype_lif.process import PrototypeLIF
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer as Source
from lava.proc.monitor.process import Monitor


class TestPrototypeLIFBitAccModel(unittest.TestCase):
    def test_s_out_bap_corresponds_to_presence_of_third_factor_in(self):
        # Params
        n_protos = 2
        t_run = 20

        # The array for 3rd factor input spike times
        s_third_factor_in = np.zeros((n_protos, t_run))

        # Inject a 3rd factor signal at the t=3, with id=1, hence targeting
        # the first prototype
        s_third_factor_in[:, 3] = [1, 1]

        # Processes
        # 3rd factor source input process (RingBuffer)
        third_factor_input = Source(data=s_third_factor_in)

        # Prototype Lif Process
        prototypes = PrototypeLIF(du=4095,
                                  dv=4095,
                                  bias_mant=0,
                                  bias_exp=0,
                                  vth=1,
                                  shape=(n_protos,),
                                  name='lif_prototypes',
                                  )

        monitor = Monitor()

        # Connections
        third_factor_input.s_out.connect(prototypes.a_third_factor_in)
        # Probe the bAP signal of the neurons
        monitor.probe(target=prototypes.s_out_bap, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        monitor.run(condition=run_cond, run_cfg=run_cfg)

        # Get results
        result = monitor.get_data()
        result = result[prototypes.name][prototypes.s_out_bap.name].T

        monitor.stop()

        # Validate the bAP signal
        expected_result = np.zeros((n_protos, t_run))
        expected_result[0, 3] = 1
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_y1_is_equal_to_third_factor_in_times_learning_rate(self):
        # Params
        n_protos = 2
        t_run = 20

        # The array for 3rd factor input spike times
        s_third_factor_in = np.zeros((n_protos, t_run))

        # Inject a 3rd factor signal at the t=3, with id=1, hence targeting
        # the first prototype
        s_third_factor_in[:, 3] = [1, 1]

        # Processes
        # 3rd factor source input process (RingBuffer)
        third_factor_input = Source(data=s_third_factor_in)

        # Prototype Lif Process
        prototypes = PrototypeLIF(du=4095,
                                  dv=4095,
                                  bias_mant=0,
                                  bias_exp=0,
                                  vth=1,
                                  shape=(n_protos,),
                                  name='lif_prototypes',
                                  )
        monitor = Monitor()

        # Connections
        third_factor_input.s_out.connect(prototypes.a_third_factor_in)

        # Probe the y1 (post-synaptic trace)
        monitor.probe(target=prototypes.s_out_y1, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        monitor.run(condition=run_cond, run_cfg=run_cfg)

        # Get the probed data
        result = monitor.get_data()
        result = result[prototypes.name][prototypes.s_out_y1.name].T

        monitor.stop()
        # Validate the post-synaptic trace: it gets updated to the value of
        # the 3rd factor signal and then stays same
        expected_result = np.zeros((n_protos, t_run))
        expected_result[0, 3:] = 127
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_neuron_outputs_spike_if_received_3rd_factor(self):
        # Params
        n_protos = 2
        t_run = 20

        # The array for 3rd factor input spike times
        s_third_factor_in = np.zeros((n_protos, t_run))

        # Inject a 3rd factor signal at the t=3, with id=1, hence targeting
        # the first prototype
        s_third_factor_in[:, 3] = [1, 1]

        # Processes
        # 3rd factor source input process (RingBuffer)
        third_factor_input = Source(data=s_third_factor_in)

        # Prototype Lif Process
        prototypes = PrototypeLIF(du=4095,
                                  dv=4095,
                                  bias_mant=0,
                                  bias_exp=0,
                                  vth=1,
                                  shape=(n_protos,),
                                  name='lif_prototypes',
                                  )
        monitor = Monitor()

        # Connections
        third_factor_input.s_out.connect(prototypes.a_third_factor_in)

        # Probe the y1 (post-synaptic trace)
        monitor.probe(target=prototypes.s_out, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        monitor.run(condition=run_cond, run_cfg=run_cfg)

        # Get the probed data
        result = monitor.get_data()
        result = result[prototypes.name][prototypes.s_out.name].T

        monitor.stop()
        # Validate the post-synaptic trace: it gets updated to the value of
        # the 3rd factor signal and then stays same
        expected_result = np.zeros((n_protos, t_run))
        expected_result[0, 3] = 1
        np.testing.assert_array_equal(result, expected_result)

    def test_neuron_vars_reset_when_reset_spike_received(self):
        # Params
        n_protos = 2
        n_features = 2
        t_run = 20
        b_fraction = 7

        # PrototypeLIF neural dynamics parameters
        du = 50
        dv = 700
        vth = 48000

        # No pattern is stored yet. None of the prototypes are allocated
        weights_proto = np.array([[0.6, 0.8], [0.8, 0.6]])
        weights_proto = weights_proto * 2 ** b_fraction

        # The graded spike array for input
        s_pattern_inp = np.zeros((n_features, t_run))
        # Original input pattern
        inp_pattern = np.array([[0.6, 0.8], [0.8, 0.6]])
        # Normalize the input pattern
        inp_pattern = inp_pattern / np.expand_dims(np.linalg.norm(
            inp_pattern, axis=1), axis=1)
        # Convert this to 8-bit fixed-point pattern
        inp_pattern = (inp_pattern * 2 ** b_fraction).astype(np.int32)
        # And inject it at the t=3
        s_pattern_inp[:, 3] = inp_pattern[0, :]
        s_pattern_inp[:, 13] = inp_pattern[1, :]

        # Processes
        data_input = Source(data=s_pattern_inp)

        # Prototype Lif Process
        prototypes = PrototypeLIF(du=du,
                                  dv=dv,
                                  bias_mant=0,
                                  bias_exp=0,
                                  vth=vth,
                                  shape=(n_protos,),
                                  name='lif_prototypes',
                                  )

        dense_proto = Dense(weights=weights_proto, num_message_bits=8)

        # WTA weights and Dense proc to be put on Prototype population
        dense_wta = Dense(weights=np.ones(shape=(n_protos, n_protos)))

        monitor = Monitor()

        # Connections
        data_input.s_out.connect(dense_proto.s_in)
        dense_proto.a_out.connect(prototypes.a_in)

        # WTA of prototypes
        prototypes.s_out.connect(dense_wta.s_in)
        dense_wta.a_out.connect(prototypes.reset_in)

        # Probe the y1 (post-synaptic trace)
        monitor.probe(target=prototypes.s_out, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        monitor.run(condition=run_cond, run_cfg=run_cfg)

        # Get the probed data
        result = monitor.get_data()
        result = result[prototypes.name][prototypes.s_out.name].T

        monitor.stop()
        # Validate the post-synaptic trace: it gets updated to the value of
        # the 3rd factor signal and then stays same
        expected_result = np.zeros((n_protos, t_run))
        expected_result[0, 7] = 1
        expected_result[1, 17] = 1

        np.testing.assert_array_equal(result, expected_result)
