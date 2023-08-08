# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.clp.novelty_detector.process import NoveltyDetector
from lava.proc.io.source import RingBuffer as Source
from lava.proc.monitor.process import Monitor


class TestNoveltyDetectorPyModel(unittest.TestCase):
    def test_detecting_novelty_in_time_window(self):
        # Params
        t_wait = 10
        t_run = 20

        # Input spikes
        spike_inp_in_aval = np.zeros((1, t_run))
        spike_inp_out_aval = np.zeros((1, t_run))

        spike_inp_in_aval[0, 3] = 1

        # Processes
        in_aval = Source(data=spike_inp_in_aval)
        out_aval = Source(data=spike_inp_out_aval)

        nvl_det = NoveltyDetector(t_wait=t_wait)
        monitor = Monitor()

        # Connections
        in_aval.s_out.connect(nvl_det.input_aval_in)
        out_aval.s_out.connect(nvl_det.output_aval_in)
        monitor.probe(target=nvl_det.novelty_detected_out, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        in_aval.run(condition=run_cond, run_cfg=run_cfg)

        result = monitor.get_data()
        result = result[nvl_det.name][nvl_det.novelty_detected_out.name].T

        in_aval.stop()
        # Validate the novelty detection output
        expected_result = np.zeros((1, t_run))
        expected_result[0, 14] = 1
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_non_activity_if_output_available(self):
        # Params
        t_wait = 10
        t_run = 20

        # Input spikes
        spike_inp_in_aval = np.zeros((1, t_run))
        spike_inp_out_aval = np.zeros((1, t_run))

        spike_inp_in_aval[0, 3] = 1
        spike_inp_out_aval[0, 7] = 1
        # Processes
        in_aval = Source(data=spike_inp_in_aval)
        out_aval = Source(data=spike_inp_out_aval)

        nvl_det = NoveltyDetector(t_wait=t_wait)
        monitor = Monitor()

        # Connections
        in_aval.s_out.connect(nvl_det.input_aval_in)
        out_aval.s_out.connect(nvl_det.output_aval_in)
        monitor.probe(target=nvl_det.novelty_detected_out, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        in_aval.run(condition=run_cond, run_cfg=run_cfg)

        result = monitor.get_data()
        result = result[nvl_det.name][nvl_det.novelty_detected_out.name].T

        in_aval.stop()
        # Validate the novelty detection output
        expected_result = np.zeros((1, t_run))
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_novelty_detection_if_output_comes_too_late(self):
        # Params
        t_wait = 10
        t_run = 20

        # Input spikes
        spike_inp_in_aval = np.zeros((1, t_run))
        spike_inp_out_aval = np.zeros((1, t_run))

        spike_inp_in_aval[0, 3] = 1
        spike_inp_out_aval[0, 15] = 1
        # Processes
        in_aval = Source(data=spike_inp_in_aval)
        out_aval = Source(data=spike_inp_out_aval)

        nvl_det = NoveltyDetector(t_wait=t_wait)
        monitor = Monitor()

        # Connections
        in_aval.s_out.connect(nvl_det.input_aval_in)
        out_aval.s_out.connect(nvl_det.output_aval_in)
        monitor.probe(target=nvl_det.novelty_detected_out, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        in_aval.run(condition=run_cond, run_cfg=run_cfg)

        result = monitor.get_data()
        result = result[nvl_det.name][nvl_det.novelty_detected_out.name].T

        in_aval.stop()
        # Validate the novelty detection output
        expected_result = np.zeros((1, t_run))
        expected_result[0, 14] = 1
        np.testing.assert_array_almost_equal(result, expected_result)
