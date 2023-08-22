# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import typing as ty
from lava.magma.core.learning.constants import GradedSpikeCfg
from lava.proc.dense.models import PyLearningDenseModelBitApproximate

from lava.utils.weightutils import SignMode

from lava.magma.core.learning.learning_rule import Loihi3FLearningRule

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.clp.prototype_lif.process import PrototypeLIF
from lava.proc.clp.novelty_detector.process import NoveltyDetector
from lava.proc.clp.nsm.process import Readout
from lava.proc.clp.nsm.process import Allocator
from lava.proc.io.source import RingBuffer
from lava.proc.monitor.process import Monitor
from lava.proc.dense.process import Dense, LearningDense


class TestPrototypesWithNoveltyDetector(unittest.TestCase):
    @staticmethod
    def create_network(t_run: int,
                       n_protos: int,
                       n_features: int,
                       weights_proto: np.ndarray,
                       inp_pattern: np.ndarray,
                       inp_times: np.ndarray) \
            -> ty.Tuple[RingBuffer, NoveltyDetector, PrototypeLIF, Dense,
                        Loihi2SimCfg, RunSteps]:
        # Params
        t_wait = 4  # Waiting window for novelty detection
        b_fraction = 8  # Fractional bits for fixed point representation

        n_protos = n_protos
        n_features = n_features
        t_run = t_run
        inp_pattern = inp_pattern  # Original input pattern
        weights_proto = weights_proto
        inp_times = inp_times  # When the input patterns should be injected

        # These are already stored patterns. Let's convert them to fixed
        # point values
        weights_proto = weights_proto * 2 ** b_fraction

        # Novelty detection input connection weights (all-to-one connections)
        weights_in_aval = np.ones(shape=(1, n_features))
        weights_out_aval = np.ones(shape=(1, n_protos))

        # The graded spike array for input
        s_pattern_inp = np.zeros((n_features, t_run))

        # Normalize the input pattern
        inp_pattern = inp_pattern / np.expand_dims(np.linalg.norm(
            inp_pattern, axis=1), axis=1)
        # Convert this to 8-bit fixed-point pattern and inject it at the t=3
        for i in range(inp_times.shape[0]):
            s_pattern_inp[:, inp_times[i]] = inp_pattern[i, :] * 2 ** b_fraction

        # Processes

        data_input = RingBuffer(data=s_pattern_inp)

        nvl_det = NoveltyDetector(t_wait=t_wait)

        allocator = Allocator(n_protos=n_protos)

        # Prototype Lif Process
        prototypes = PrototypeLIF(du=4095,
                                  dv=4095,
                                  bias_mant=0,
                                  bias_exp=0,
                                  vth=64000,
                                  shape=(n_protos,),
                                  name='lif_prototypes',
                                  )

        dense_proto = Dense(weights=weights_proto, num_message_bits=8)
        dense_in_aval = Dense(weights=weights_in_aval)
        dense_out_aval = Dense(weights=weights_out_aval)
        dense_3rd_factor = Dense(weights=np.ones(shape=(n_protos, 1)),
                                 num_message_bits=8)

        # Connections

        data_input.s_out.connect(dense_proto.s_in)
        dense_proto.a_out.connect(prototypes.a_in)

        data_input.s_out.connect(dense_in_aval.s_in)
        dense_in_aval.a_out.connect(nvl_det.input_aval_in)

        prototypes.s_out.connect(dense_out_aval.s_in)
        dense_out_aval.a_out.connect(nvl_det.output_aval_in)

        # Novelty detector -> Allocator -> Dense -> PrototypeLIF connection
        nvl_det.novelty_detected_out.connect(allocator.trigger_in)
        allocator.allocate_out.connect(dense_3rd_factor.s_in)
        dense_3rd_factor.a_out.connect(prototypes.a_third_factor_in)

        exception_map = {
            LearningDense: PyLearningDenseModelBitApproximate
        }
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt",
                               exception_proc_model_map=exception_map)

        return data_input, nvl_det, prototypes, dense_proto, run_cfg, run_cond

    def test_detecting_novelty_if_no_match(self):
        t_run = 10
        n_protos = 2
        n_features = 2
        weights_proto = np.array([[0.6, 0.8], [0, 0]])
        inp_pattern = np.array([[0.82, 0.55]])
        inp_times = np.array([3])

        _, nvl_det, prototypes, _, run_cfg, run_cond = \
            self.create_network(t_run, n_protos, n_features, weights_proto,
                                inp_pattern, inp_times)

        monitor = Monitor()

        monitor.probe(target=nvl_det.novelty_detected_out, num_steps=t_run)

        # Run
        prototypes.run(condition=run_cond, run_cfg=run_cfg)

        # Get results
        result = monitor.get_data()
        result = result[nvl_det.name][nvl_det.novelty_detected_out.name].T

        prototypes.stop()

        # Validate the bAP signal and y1 trace
        expected_result = np.zeros((1, t_run))
        expected_result[0, 9] = 1
        np.testing.assert_array_equal(result, expected_result)

    def test_two_consecutive_novelty_detection(self):
        # Params
        t_run = 20
        n_protos = 2
        n_features = 2
        weights_proto = np.array([[0, 0], [0, 0]])
        inp_pattern = np.array([[0.82, 0.55], [0.55, 0.82]])
        inp_times = np.array([3, 13])

        _, nvl_det, prototypes, _, run_cfg, run_cond = \
            self.create_network(t_run, n_protos, n_features, weights_proto,
                                inp_pattern, inp_times)

        monitor = Monitor()

        monitor.probe(target=nvl_det.novelty_detected_out, num_steps=t_run)

        # Run
        prototypes.run(condition=run_cond, run_cfg=run_cfg)

        # Get results
        result = monitor.get_data()
        result = result[nvl_det.name][nvl_det.novelty_detected_out.name].T

        prototypes.stop()

        # Validate the bAP signal and y1 trace
        expected_result = np.zeros((1, t_run))
        expected_result[0, 9] = 1
        expected_result[0, 19] = 1
        np.testing.assert_array_equal(result, expected_result)

    def test_novelty_signal_is_correctly_received_by_prototypes(self):
        # Params
        t_run = 12
        n_protos = 2
        n_features = 2
        weights_proto = np.array([[0, 0], [0, 0]])
        inp_pattern = np.array([[0.82, 0.55]])
        inp_times = np.array([3])

        _, _, prototypes, _, run_cfg, run_cond = \
            self.create_network(t_run, n_protos, n_features, weights_proto,
                                inp_pattern, inp_times)

        # Monitors
        monitor_bap = Monitor()
        monitor_y1 = Monitor()

        monitor_bap.probe(target=prototypes.s_out_bap, num_steps=t_run)
        monitor_y1.probe(target=prototypes.s_out_y1, num_steps=t_run)

        # Run
        prototypes.run(condition=run_cond, run_cfg=run_cfg)

        # Get results
        result_bap = monitor_bap.get_data()
        result_bap = result_bap[prototypes.name][prototypes.s_out_bap.name].T

        result_y1 = monitor_y1.get_data()
        result_y1 = result_y1[prototypes.name][prototypes.s_out_y1.name].T

        prototypes.stop()

        # Validate the bAP signal and y1 trace
        expected_bap = np.zeros((n_protos, t_run))
        expected_bap[0, 10] = 1
        np.testing.assert_array_equal(result_bap, expected_bap)

        expected_y1 = np.zeros((n_protos, t_run))
        expected_y1[0, [10, 11]] = [127, 127]
        np.testing.assert_array_equal(result_y1, expected_y1)

    def test_recognize_stored_patterns(self):
        # Params
        t_run = 20
        n_protos = 2
        n_features = 2
        weights_proto = np.array([[0.8, 0.6], [0.6, 0.8]])
        inp_pattern = np.array([[0.82, 0.55], [0.55, 0.82]])
        inp_times = np.array([3, 13])

        _, nvl_det, prototypes, _, run_cfg, run_cond = \
            self.create_network(t_run, n_protos, n_features, weights_proto,
                                inp_pattern, inp_times)

        monitor_nvl = Monitor()
        monitor_protos = Monitor()

        monitor_nvl.probe(target=nvl_det.novelty_detected_out, num_steps=t_run)
        monitor_protos.probe(target=prototypes.s_out, num_steps=t_run)

        # Run
        prototypes.run(condition=run_cond, run_cfg=run_cfg)

        # Get results
        result_nvl = monitor_nvl.get_data()
        result_nvl = result_nvl[nvl_det.name][
            nvl_det.novelty_detected_out.name].T

        result_protos = monitor_protos.get_data()
        result_protos = result_protos[prototypes.name][prototypes.s_out.name].T

        prototypes.stop()

        # Validate the output of the prototype neurons.
        expected_proto_out = np.zeros((n_protos, t_run))
        expected_proto_out[0, 4] = 1
        expected_proto_out[1, 14] = 1
        np.testing.assert_array_equal(result_protos, expected_proto_out)

        # Validate the novelty detection output. In this case no novelty
        # should be detected as the tested patterns are already stored in the
        # prototype weights
        expected_nvl = np.zeros((1, t_run))
        np.testing.assert_array_equal(result_nvl, expected_nvl)


class TestOneShotLearning(unittest.TestCase):

    def test_nvl_detection_triggers_one_shot_learning(self):
        # General params
        t_wait = 4
        n_protos = 3
        n_features = 2
        b_fraction = 7
        t_run = 25

        # LIF parameters
        du = 4095
        dv = 4095
        vth = 63000

        # Trace decay constants
        x1_tau = 65535

        # Epoch length
        t_epoch = 1

        # No pattern is stored yet. None of the prototypes are allocated
        weights_proto = np.array([[0, 0], [0, 0], [0, 0]])
        weights_proto = weights_proto * 2 ** b_fraction

        # Config for Writing graded payload to x1-trace
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        # Novelty detection input connection weights  (all-to-one
        # connections)
        weights_in_aval = np.ones(shape=(1, n_features))
        weights_out_aval = np.ones(shape=(1, n_protos))

        # The graded spike array for input
        s_pattern_inp = np.zeros((n_features, t_run))
        # Original input pattern
        inp_pattern = np.array([[0.78, 0.58], [0.59, 0.81]])
        # Normalize the input pattern
        inp_pattern = inp_pattern / np.expand_dims(np.linalg.norm(
            inp_pattern, axis=1), axis=1)
        # Convert this to 8-bit fixed-point pattern
        inp_pattern = (inp_pattern * 2 ** b_fraction).astype(np.int32)
        # and inject it at the t=3
        s_pattern_inp[:, 3] = inp_pattern[0, :]
        s_pattern_inp[:, 13] = inp_pattern[1, :]

        # Create custom LearningRule. Define dw as string
        dw = "2^-3*y1*x1*y0"

        learning_rule = Loihi3FLearningRule(dw=dw,
                                            x1_tau=x1_tau,
                                            t_epoch=t_epoch,)

        # Processes
        data_input = RingBuffer(data=s_pattern_inp)

        nvl_det = NoveltyDetector(t_wait=t_wait)

        allocator = Allocator(n_protos=n_protos)

        # Prototype Lif Process
        prototypes = PrototypeLIF(du=du,
                                  dv=dv,
                                  bias_mant=0,
                                  bias_exp=0,
                                  vth=vth,
                                  shape=(n_protos,),
                                  name='lif_prototypes',
                                  sign_mode=SignMode.EXCITATORY,
                                  learning_rule=learning_rule)

        dense_proto = LearningDense(weights=weights_proto,
                                    learning_rule=learning_rule,
                                    name="proto_weights",
                                    num_message_bits=8,
                                    graded_spike_cfg=graded_spike_cfg)

        dense_in_aval = Dense(weights=weights_in_aval)
        dense_out_aval = Dense(weights=weights_out_aval)
        dense_3rd_factor = Dense(weights=np.ones(shape=(n_protos, 1)),
                                 num_message_bits=8)

        monitor_nvl = Monitor()
        monitor_weights = Monitor()
        monitor_x1_trace = Monitor()

        # Connections

        data_input.s_out.connect(dense_proto.s_in)
        dense_proto.a_out.connect(prototypes.a_in)

        data_input.s_out.connect(dense_in_aval.s_in)
        dense_in_aval.a_out.connect(nvl_det.input_aval_in)

        prototypes.s_out.connect(dense_out_aval.s_in)
        dense_out_aval.a_out.connect(nvl_det.output_aval_in)

        # Novelty detector -> Allocator -> Dense -> PrototypeLIF connection
        nvl_det.novelty_detected_out.connect(allocator.trigger_in)
        allocator.allocate_out.connect(dense_3rd_factor.s_in)
        dense_3rd_factor.a_out.connect(prototypes.a_third_factor_in)

        prototypes.s_out_bap.connect(dense_proto.s_in_bap)

        # Sending y1 spike
        prototypes.s_out_y1.connect(dense_proto.s_in_y1)

        # Probe novelty detector and prototypes
        monitor_nvl.probe(target=nvl_det.novelty_detected_out, num_steps=t_run)
        monitor_x1_trace.probe(target=dense_proto.x1, num_steps=t_run)
        monitor_weights.probe(target=dense_proto.weights, num_steps=t_run)

        # Run
        exception_map = {
            LearningDense: PyLearningDenseModelBitApproximate
        }
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt",
                               exception_proc_model_map=exception_map)

        prototypes.run(condition=run_cond, run_cfg=run_cfg)

        # Get results
        result_nvl = monitor_nvl.get_data()
        result_nvl = result_nvl[nvl_det.name][
            nvl_det.novelty_detected_out.name].T

        result_x1_trace = monitor_x1_trace.get_data()['proto_weights']['x1'].T

        result_weights = monitor_weights.get_data()
        result_weights = result_weights[dense_proto.name][
            dense_proto.weights.name].T

        # Stop the run
        prototypes.stop()

        # Do the tests
        expected_nvl = np.zeros((1, t_run))
        expected_nvl[0, 9] = 1
        expected_nvl[0, 19] = 1

        exp_x1_0 = np.ceil(inp_pattern[0, :] / 2)
        exp_x1_1 = np.ceil(inp_pattern[1, :] / 2)

        expected_x1 = np.zeros((n_features, t_run))
        expected_x1[:, 3:13] = np.tile(exp_x1_0[:, None], 10)
        expected_x1[:, 13:] = np.tile(exp_x1_1[:, None], 12)

        exp_w_0 = (exp_x1_0 - 1) * 2
        exp_w_1 = (exp_x1_1 - 1) * 2
        expected_weights = np.zeros((n_features, n_protos, t_run))

        expected_weights[:, 0, 10:] = np.tile(exp_w_0[:, None], t_run - 10)
        expected_weights[:, 1, 20:] = np.tile(exp_w_1[:, None], t_run - 20)

        np.testing.assert_array_equal(result_nvl, expected_nvl)
        # np.testing.assert_array_equal(expected_x1, result_x1_trace)
        np.testing.assert_array_almost_equal(expected_weights, result_weights,
                                             decimal=-1)

    def test_allocation_triggered_by_erroneous_classification(self):
        # General params
        t_wait = 4
        n_protos = 3
        n_features = 2
        b_fraction = 8
        t_run = 33

        # LIF parameters
        du = 4095
        dv = 4095
        vth = 63000

        # Trace decay constants
        x1_tau = 65535

        # Epoch length
        t_epoch = 1

        # No pattern is stored yet. None of the prototypes are allocated
        weights_proto = np.array([[0, 0], [0, 0], [0, 0]])
        weights_proto = weights_proto * 2 ** b_fraction

        # Config for Writing graded payload to x1-trace
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        # Novelty detection input connection weights  (all-to-one
        # connections)
        weights_in_aval = np.ones(shape=(1, n_features))
        weights_out_aval = np.ones(shape=(1, n_protos))

        # The graded spike array for input
        s_pattern_inp = np.zeros((n_features, t_run))
        # Original input pattern
        inp_pattern = np.array([[0.82, 0.55], [0.55, 0.82], [0.87, 0.50]])
        # Normalize the input pattern
        inp_pattern = inp_pattern / np.expand_dims(np.linalg.norm(
            inp_pattern, axis=1), axis=1)
        # Convert this to 8-bit fixed-point pattern
        inp_pattern = (inp_pattern * 2 ** b_fraction).astype(np.int32)
        # and inject it at the t=3
        s_pattern_inp[:, 3] = inp_pattern[0, :]
        s_pattern_inp[:, 13] = inp_pattern[1, :]
        s_pattern_inp[:, 23] = inp_pattern[2, :]

        # The graded spike array for the user-provided label
        s_user_label = np.zeros((1, t_run))
        s_user_label[0, 9] = 1
        s_user_label[0, 19] = 2
        s_user_label[0, 29] = 3

        # Create custom LearningRule. Define dw as string
        dw = "2^-3*y1*x1*y0"

        learning_rule = Loihi3FLearningRule(dw=dw,
                                            x1_tau=x1_tau,
                                            t_epoch=t_epoch)

        # Processes
        data_input = RingBuffer(data=s_pattern_inp)

        nvl_det = NoveltyDetector(t_wait=t_wait)

        allocator = Allocator(n_protos=n_protos)

        readout = Readout(n_protos=n_protos)

        label_in = RingBuffer(data=s_user_label)

        # Prototype Lif Process
        prototypes = PrototypeLIF(du=du,
                                  dv=dv,
                                  bias_mant=0,
                                  bias_exp=0,
                                  vth=vth,
                                  shape=(n_protos,),
                                  name='lif_prototypes',
                                  sign_mode=SignMode.EXCITATORY,
                                  learning_rule=learning_rule)

        dense_proto = LearningDense(weights=weights_proto,
                                    learning_rule=learning_rule,
                                    name="proto_weights",
                                    num_message_bits=8,
                                    graded_spike_cfg=graded_spike_cfg)

        dense_in_aval = Dense(weights=weights_in_aval)
        dense_out_aval = Dense(weights=weights_out_aval)
        dense_alloc_weight = Dense(weights=np.ones(shape=(1, 1)))
        dense_3rd_factor = Dense(weights=np.ones(shape=(n_protos, 1)),
                                 num_message_bits=8)

        monitor_nvl = Monitor()
        monitor_protos = Monitor()
        monitor_alloc = Monitor()

        # Connections

        data_input.s_out.connect(dense_proto.s_in)
        dense_proto.a_out.connect(prototypes.a_in)

        data_input.s_out.connect(dense_in_aval.s_in)
        dense_in_aval.a_out.connect(nvl_det.input_aval_in)

        prototypes.s_out.connect(dense_out_aval.s_in)
        dense_out_aval.a_out.connect(nvl_det.output_aval_in)

        # Novelty detector -> Allocator -> Dense -> PrototypeLIF connection
        nvl_det.novelty_detected_out.connect(allocator.trigger_in)
        allocator.allocate_out.connect(dense_3rd_factor.s_in)
        dense_3rd_factor.a_out.connect(prototypes.a_third_factor_in)

        prototypes.s_out_bap.connect(dense_proto.s_in_bap)

        # Sending y1 spike
        prototypes.s_out_y1.connect(dense_proto.s_in_y1)

        # Prototype Neurons' outputs connect to the inference input of the
        # Readout process
        prototypes.s_out.connect(readout.inference_in)

        # Label input to the Readout proces
        label_in.s_out.connect(readout.label_in)

        # Readout trigger to the Allocator
        readout.trigger_alloc.connect(dense_alloc_weight.s_in)
        dense_alloc_weight.a_out.connect(allocator.trigger_in)

        # Probe novelty detector and prototypes
        monitor_nvl.probe(target=nvl_det.novelty_detected_out, num_steps=t_run)
        monitor_protos.probe(target=prototypes.s_out, num_steps=t_run)
        monitor_alloc.probe(target=allocator.allocate_out, num_steps=t_run)

        # Run
        exception_map = {
            LearningDense: PyLearningDenseModelBitApproximate
        }
        run_cond = RunSteps(num_steps=t_run)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt",
                               exception_proc_model_map=exception_map)

        prototypes.run(condition=run_cond, run_cfg=run_cfg)

        # Get results
        result_nvl = monitor_nvl.get_data()
        result_nvl = result_nvl[nvl_det.name][
            nvl_det.novelty_detected_out.name].T

        result_protos = monitor_protos.get_data()
        result_protos = result_protos[prototypes.name][prototypes.s_out.name].T

        result_alloc = monitor_alloc.get_data()
        result_alloc = result_alloc[allocator.name][
            allocator.allocate_out.name].T

        # Stop the run
        prototypes.stop()

        # Do the tests
        expected_nvl = np.zeros((1, t_run))
        expected_nvl[0, [9, 19]] = [1, 1]

        expected_alloc = np.zeros((1, t_run))
        expected_alloc[0, 9] = 1
        expected_alloc[0, 19] = 2
        expected_alloc[0, 30] = 3

        expected_proto_out = np.zeros((n_protos, t_run))
        # 1) novelty-based allocation triggered, 2) erroneous prediction
        expected_proto_out[0, [10, 24]] = 1
        expected_proto_out[1, 20] = 1  # Novelty-based allocation triggered
        expected_proto_out[2, 31] = 1  # Error-based allocation triggered

        np.testing.assert_array_equal(result_nvl, expected_nvl)
        np.testing.assert_array_equal(result_alloc, expected_alloc)
        np.testing.assert_array_equal(result_protos, expected_proto_out)
