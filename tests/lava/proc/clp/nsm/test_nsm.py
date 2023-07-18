# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.clp.nsm.process import Readout
from lava.proc.clp.nsm.process import Allocator
from lava.proc.io.source import RingBuffer as Source
from lava.proc.monitor.process import Monitor


class TestReadoutPyModel(unittest.TestCase):
    def test_pseudo_labeling_if_no_user_label(self):
        # Params
        n_protos = 2
        t_run = 20

        # Input spikes
        s_infer_in = np.zeros((n_protos, t_run))
        s_label_in = np.zeros((1, t_run))

        # Prototype 0 is active at t=3
        s_infer_in[0, 3] = 1

        # Processes
        infer_in = Source(data=s_infer_in)
        label_in = Source(data=s_label_in)

        # Readout process is initialized without any labels, so they are all
        # zero
        readout_layer = Readout(n_protos=n_protos)
        monitor = Monitor()

        # Connections
        infer_in.s_out.connect(readout_layer.inference_in)
        label_in.s_out.connect(readout_layer.label_in)
        monitor.probe(target=readout_layer.user_output, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        infer_in.run(condition=run_cond, run_cfg=run_cfg)

        result = monitor.get_data()
        result = result[readout_layer.name][readout_layer.user_output.name].T

        infer_in.stop()
        # Validate the user outputs
        expected_result = np.zeros((1, t_run))
        expected_result[0, 3] = -1  # we expect a pseudo-label
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_readout_from_labelled_winner(self):
        # Params
        n_protos = 2
        t_run = 20

        # Input spikes
        s_infer_in = np.zeros((n_protos, t_run))
        s_label_in = np.zeros((1, t_run))

        # Prototype 0 is active at t=3
        s_infer_in[0, 3] = 1

        # Processes
        infer_in = Source(data=s_infer_in)
        label_in = Source(data=s_label_in)

        # Prototype 0 has the label "1", Prototype 1 has no label, because
        # the proto_labels are initialized with [1,0]
        readout_layer = Readout(n_protos=n_protos,
                                proto_labels=np.array([1, 0]))
        monitor = Monitor()

        # Connections
        infer_in.s_out.connect(readout_layer.inference_in)
        label_in.s_out.connect(readout_layer.label_in)
        monitor.probe(target=readout_layer.user_output, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        infer_in.run(condition=run_cond, run_cfg=run_cfg)

        result = monitor.get_data()
        result = result[readout_layer.name][readout_layer.user_output.name].T

        infer_in.stop()
        # Validate the user outputs
        expected_result = np.zeros((1, t_run))
        expected_result[0, 3] = 1  # We expect the label "1"
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_labelling_a_pseudo_labelled_winner(self):
        # Params
        n_protos = 2
        t_run = 20

        # Input spikes
        s_infer_in = np.zeros((n_protos, t_run))
        s_label_in = np.zeros((1, t_run))

        s_infer_in[0, 3] = 1  # Winner is the 0'th prototype
        s_label_in[0, 6] = 2  # Label provided by the user

        # Processes
        infer_in = Source(data=s_infer_in)
        label_in = Source(data=s_label_in)

        readout_layer = Readout(n_protos=n_protos,
                                proto_labels=np.array([-1, 0]))
        monitor = Monitor()

        # Connections
        infer_in.s_out.connect(readout_layer.inference_in)
        label_in.s_out.connect(readout_layer.label_in)
        monitor.probe(target=readout_layer.user_output, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        infer_in.run(condition=run_cond, run_cfg=run_cfg)

        result = monitor.get_data()
        result = result[readout_layer.name][readout_layer.user_output.name].T

        infer_in.stop()
        # Validate the novelty detection output
        expected_result = np.zeros((1, t_run))

        # The predicted label is first -1, but then it is updated by the
        # user-provided label
        expected_result[0, 3] = -1
        expected_result[0, 6] = 2

        np.testing.assert_array_almost_equal(result, expected_result)

    def test_feedback_and_allocation_output(self):
        # Params
        n_protos = 2
        t_run = 20

        # Input spikes
        s_infer_in = np.zeros((n_protos, t_run))
        s_label_in = np.zeros((1, t_run))

        s_infer_in[0, [3, 12]] = 1
        s_label_in[0, 6] = 2  # Label that match the inference
        s_label_in[0, 17] = 1  # Label that does not match the inference
        # Processes
        infer_in = Source(data=s_infer_in)
        label_in = Source(data=s_label_in)

        readout_layer = Readout(n_protos=n_protos,
                                proto_labels=np.array([2, 0]))
        monitor_fb = Monitor()
        monitor_alloc = Monitor()

        # Connections
        infer_in.s_out.connect(readout_layer.inference_in)
        label_in.s_out.connect(readout_layer.label_in)

        monitor_fb.probe(target=readout_layer.feedback, num_steps=t_run)
        monitor_alloc.probe(target=readout_layer.trigger_alloc, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        infer_in.run(condition=run_cond, run_cfg=run_cfg)

        result_fb = monitor_fb.get_data()
        result_fb = result_fb[readout_layer.name][readout_layer.feedback.name].T

        result_alloc = monitor_alloc.get_data()
        result_alloc = result_alloc[readout_layer.name][
            readout_layer.trigger_alloc.name].T

        infer_in.stop()
        # Validate the novelty detection output
        expected_fb = np.zeros(shape=(1, t_run))
        expected_fb[0, 6] = 1  # We expect a match for the first input
        expected_fb[0, 17] = -1  # We expect mismatch for the second input
        np.testing.assert_array_equal(result_fb, expected_fb)

        expected_alloc = np.zeros(shape=(1, t_run))
        # We expect allocation trigger output when there is a mismatch
        expected_alloc[0, 17] = 1
        np.testing.assert_array_equal(result_alloc, expected_alloc)


class TestAllocatorPyModel(unittest.TestCase):
    def test_allocator_output(self):
        # Params
        n_protos = 4
        t_run = 10

        # Input spikes
        s_trigger_in = np.zeros((1, t_run))

        s_trigger_in[0, [3, 7]] = [1, 1]  # Trigger input
        # Processes
        alloc_trigger_in = Source(data=s_trigger_in)

        allocator = Allocator(n_protos=n_protos)
        monitor_alloc = Monitor()

        # Connections
        alloc_trigger_in.s_out.connect(allocator.trigger_in)

        monitor_alloc.probe(target=allocator.allocate_out, num_steps=t_run)

        # Run
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        allocator.run(condition=run_cond, run_cfg=run_cfg)

        result_alloc = monitor_alloc.get_data()
        result_alloc = result_alloc[allocator.name][
            allocator.allocate_out.name].T

        allocator.stop()

        # Validate the allocation output
        expected_alloc = np.zeros(shape=(1, t_run))
        expected_alloc[0, 3] = 1
        expected_alloc[0, 7] = 2
        np.testing.assert_array_equal(result_alloc, expected_alloc)
