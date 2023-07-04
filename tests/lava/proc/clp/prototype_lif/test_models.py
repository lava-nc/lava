# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.clp.prototype_lif.process import PrototypeLIF
from lava.proc.io.source import RingBuffer as Source
from lava.proc.monitor.process import Monitor


class TestPrototypeLIFBitAccModel(unittest.TestCase):
    def test_s_out_bap_corresponds_to_presence_of_third_factor_in(self):
        # Params
        n_protos = 2
        t_run = 20

        # The array for 3rd factor input spike times
        s_third_factor_in = np.zeros((n_protos, t_run))

        # Inject a 3rd factor signal at the t=3
        s_third_factor_in[0, 3] = 127

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

        # Inject a 3rd factor signal at the t=3
        s_third_factor_in[0, 3] = 1

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

        # Inject a 3rd factor signal at the t=3
        s_third_factor_in[0, 3] = 127

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
