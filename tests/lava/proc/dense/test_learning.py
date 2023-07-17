# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import typing as ty

from lava.magma.core.learning.constants import GradedSpikeCfg
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.dense.process import LearningDense
from lava.proc.io.source import RingBuffer, PySendModelFixed, PySendModelFloat


class TestLearningSimGradedSpikeFloatingPoint(unittest.TestCase):
    @staticmethod
    def create_network(num_steps: int,
                       learning_rule_cnd: str,
                       graded_spike_cfg: GradedSpikeCfg) \
            -> ty.Tuple[RingBuffer, LearningDense, RingBuffer]:
        """Create a network of RingBuffer -> LearningDense -> RingBuffer.

        Parameters
        ----------
        num_steps : int
            Time steps size for RingBuffer.
        learning_rule_cnd : str
            String specifying which learning rule condition to use.
        graded_spike_cfg : GradedSpikeCfg
            GradedSpikeCfg to use for the LearningDense.

        Returns
        ----------
        pre_ring_buffer : RingBuffer
            Pre-synaptic RingBuffer Process.
        learning_dense : LearningDense
            LearningDense Process.
        post_ring_buffer : RingBuffer
            Post-synaptic RingBuffer Process.
        """

        dw = f"{learning_rule_cnd} * x1"

        learning_rule = \
            Loihi2FLearningRule(dw=dw,
                                x1_impulse=16, x1_tau=20,
                                x2_impulse=24, x2_tau=5,
                                t_epoch=4)

        # Pre-synaptic spike at time step 2, payload 51
        spike_raster_pre = np.zeros((1, num_steps))
        spike_raster_pre[0, 1] = 51
        # Post-synaptic spike at time step 3
        spike_raster_post = np.zeros((1, num_steps))
        spike_raster_post[0, 2] = 1

        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))
        learning_dense = \
            LearningDense(weights=np.eye(1) * 10,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=graded_spike_cfg)
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        return pattern_pre, learning_dense, pattern_post

    def setUp(self) -> None:
        exception_map = {
            RingBuffer: PySendModelFloat
        }
        self._run_cfg = \
            Loihi2SimCfg(select_tag="floating_pt",
                         exception_proc_model_map=exception_map)
        self._run_cnd = RunSteps(num_steps=1)

    def test_learning_graded_spike_reg_imp_float_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "x0"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # x1 updates with decayed regular impulse, at the end of the epoch
        expected_x1_data = [0.0, 0.0, 0.0, 14.4773986, 14.4773986]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        expected_wgt_data = [10.0, 10.0, 10.0, 26.0, 26.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_reg_imp_float_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=y0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "y0"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # x1 updates with decayed regular impulse, at the end of the epoch
        expected_x1_data = [0.0, 0.0, 0.0, 14.4773986, 14.4773986]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at post-spike time
        # i.e 1 time step after pre-spike time, hence the decay
        expected_wgt_data = [10.0, 10.0, 10.0, 25.2196707, 25.2196707]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_reg_imp_float_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=u0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "u0"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # x1 updates with decayed regular impulse, at the end of the epoch
        expected_x1_data = [0.0, 0.0, 0.0, 14.4773986, 14.4773986]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at post-spike time
        # i.e 2 time step after pre-spike time, hence the decay
        expected_wgt_data = [10.0, 10.0, 10.0, 24.4773986, 24.4773986]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_float_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=x0 * x1
        learning rule."""

        num_steps = 5

        learning_rule_cnd = "x0"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of overwrite
        learning_dense.x1.init = np.array([15.0])

        # graded spike payload/2 overwrites x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 25.5, 25.5, 20.8776342, 20.8776342]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at pre-spike time
        # is the value of x1 decayed by 2 time steps
        expected_wgt_data = [10.0, 10.0, 10.0, 33.0733541, 33.0733541]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_float_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=y0 * x1
        learning rule."""

        num_steps = 5

        learning_rule_cnd = "y0"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of overwrite
        learning_dense.x1.init = np.array([15.0])

        # graded spike payload/2 overwrites x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 25.5, 25.5, 20.8776342, 20.8776342]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at post-spike time
        # is the value of x1 decayed by 3 time steps
        expected_wgt_data = [10.0, 10.0, 10.0, 31.9480533, 31.9480533]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_float_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=u0 * x1
        learning rule."""

        num_steps = 5

        learning_rule_cnd = "u0"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of overwrite
        learning_dense.x1.init = np.array([15.0])

        # graded spike payload/2 overwrites x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 25.5, 25.5, 20.8776342, 20.8776342]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at the end of this epoch
        # is the value of x1 decayed by 4 time steps (t_epoch)
        expected_wgt_data = [10.0, 10.0, 10.0, 30.8776342, 30.8776342]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_float_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITH_SATURATION GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "x0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITH_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of addition
        learning_dense.x1.init = np.array([15.0])

        # graded spike payload/2 is added to x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 40.5, 40.5, 33.1585954, 33.1585954]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0, 0, 0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at pre-spike time
        # is the value of x1 decayed by 2 time steps
        expected_wgt_data = [10.0, 10.0, 10.0, 46.6459154, 46.6459154]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_float_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITH_SATURATION GradedSpikeCfg and
        dw=y0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "y0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITH_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of addition
        learning_dense.x1.init = np.array([15.0])

        # graded spike payload/2 is added to x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 40.5, 40.5, 33.1585954, 33.1585954]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0, 0, 0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at post-spike time
        # is the value of x1 decayed by 3 time steps
        expected_wgt_data = [10.0, 10.0, 10.0, 44.8586730, 44.8586730]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_float_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITH_SATURATION GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "u0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITH_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of overwrite
        learning_dense.x1.init = np.array([15.0])

        # graded spike payload/2 is added to x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 40.5, 40.5, 33.1585954, 33.1585954]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0, 0, 0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at the end of this epoch
        # is the value of x1 decayed by 4 time steps (t_epoch)
        expected_wgt_data = [10.0, 10.0, 10.0, 43.1585954, 43.1585954]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_float_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITHOUT_SATURATION GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "x0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITHOUT_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of addition
        learning_dense.x1.init = np.array([15.0])

        # graded spike payload/2 is added to x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 40.5, 40.5, 33.1585954, 33.1585954]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0, 0, 0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at pre-spike time
        # is the value of x1 decayed by 2 time steps
        expected_wgt_data = [10.0, 10.0, 10.0, 46.6459154, 46.6459154]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_float_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITHOUT_SATURATION GradedSpikeCfg and
        dw=y0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "y0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITHOUT_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of addition
        learning_dense.x1.init = np.array([15.0])

        # graded spike payload/2 is added to x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 40.5, 40.5, 33.1585954, 33.1585954]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0, 0, 0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at post-spike time
        # is the value of x1 decayed by 3 time steps
        expected_wgt_data = [10.0, 10.0, 10.0, 44.8586730, 44.8586730]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_float_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITHOUT_SATURATION GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "u0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITHOUT_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of overwrite
        learning_dense.x1.init = np.array([15.0])

        # graded spike payload/2 is added to x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 40.5, 40.5, 33.1585954, 33.1585954]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0, 0, 0, 16.0876811, 16.0876811]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at the end of this epoch
        # is the value of x1 decayed by 4 time steps (t_epoch)
        expected_wgt_data = [10.0, 10.0, 10.0, 43.1585954, 43.1585954]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)


class TestLearningSimGradedSpikeBitApprox(unittest.TestCase):
    @staticmethod
    def create_network(num_steps: int,
                       learning_rule_cnd: str,
                       graded_spike_cfg: GradedSpikeCfg) \
            -> ty.Tuple[RingBuffer, LearningDense, RingBuffer]:
        """Create a network of RingBuffer -> LearningDense -> RingBuffer.

        Parameters
        ----------
        num_steps : int
            Time steps size for RingBuffer.
        learning_rule_cnd : str
            String specifying which learning rule condition to use.
        graded_spike_cfg : GradedSpikeCfg
            GradedSpikeCfg to use for the LearningDense.

        Returns
        ----------
        pre_ring_buffer : RingBuffer
            Pre-synaptic RingBuffer Process.
        learning_dense : LearningDense
            LearningDense Process.
        post_ring_buffer : RingBuffer
            Post-synaptic RingBuffer Process.
        """

        dw = f"{learning_rule_cnd} * x1"

        learning_rule = \
            Loihi2FLearningRule(dw=dw,
                                x1_impulse=16, x1_tau=20,
                                x2_impulse=24, x2_tau=5,
                                t_epoch=4, rng_seed=0)

        # Pre-synaptic spike at time step 2, payload 51
        spike_raster_pre = np.zeros((1, num_steps))
        spike_raster_pre[0, 1] = 51
        # Post-synaptic spike at time step 3
        spike_raster_post = np.zeros((1, num_steps))
        spike_raster_post[0, 2] = 1

        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))
        learning_dense = \
            LearningDense(weights=np.eye(1) * 10,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=graded_spike_cfg)
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        return pattern_pre, learning_dense, pattern_post

    def setUp(self) -> None:
        exception_map = {
            RingBuffer: PySendModelFixed
        }
        self._run_cfg = \
            Loihi2SimCfg(select_tag="bit_approximate_loihi",
                         exception_proc_model_map=exception_map)
        self._run_cnd = RunSteps(num_steps=1)

    def test_learning_graded_spike_reg_imp_bit_approx_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "x0"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # x1 updates with decayed regular impulse, at the end of the epoch
        expected_x1_data = [0.0, 0.0, 0.0, 14.0, 14.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        expected_wgt_data = [10.0, 10.0, 10.0, 26.0, 26.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_reg_imp_bit_approx_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=y0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "y0"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pre_ring_buffer, dense, _ = self.create_network(num_steps,
                                                        learning_rule_cnd,
                                                        graded_spike_cfg)

        # x1 updates with decayed regular impulse, at the end of the epoch
        expected_x1_data = [0.0, 0.0, 0.0, 14.0, 14.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0]
        # weight update at the end of the epoch = value of x1 at post-spike time
        # i.e 1 time step after pre-spike time, hence the decay
        expected_wgt_data = [10.0, 10.0, 10.0, 25.0, 25.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(dense.x1.get()[0])
            x2_data.append(dense.x2.get()[0])
            wgt_data.append(dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_reg_imp_bit_approx_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, USE_REGULAR_IMPULSE GradedSpikeCfg and
        dw=u0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "u0"
        graded_spike_cfg = GradedSpikeCfg.USE_REGULAR_IMPULSE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # x1 updates with decayed regular impulse, at the end of the epoch
        expected_x1_data = [0.0, 0.0, 0.0, 14.0, 14.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0]
        # weight update at the end of the epoch = value of x1 at post-spike time
        # i.e 2 time step after pre-spike time, hence the decay
        expected_wgt_data = [10.0, 10.0, 10.0, 24.0, 24.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_bit_approx_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=x0 * x1
        learning rule."""

        num_steps = 5

        learning_rule_cnd = "x0"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of overwrite
        learning_dense.x1.init = np.array([15], dtype=int)

        # graded spike payload/2 overwrites x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 26.0, 26.0, 22.0, 22.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at pre-spike time
        # is the value of x1 decayed by 2 time steps
        expected_wgt_data = [10.0, 10.0, 10.0, 34.0, 34.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_bit_approx_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=y0 * x1
        learning rule."""

        num_steps = 5

        learning_rule_cnd = "y0"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of overwrite
        learning_dense.x1.init = np.array([15], dtype=int)

        # graded spike payload/2 overwrites x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 26.0, 26.0, 22.0, 22.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at post-spike time
        # is the value of x1 decayed by 3 time steps
        expected_wgt_data = [10.0, 10.0, 10.0, 33.0, 33.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_overwrite_bit_approx_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, OVERWRITE GradedSpikeCfg and dw=u0 * x1
        learning rule."""

        num_steps = 5

        learning_rule_cnd = "u0"
        graded_spike_cfg = GradedSpikeCfg.OVERWRITE

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of overwrite
        learning_dense.x1.init = np.array([15], dtype=int)

        # graded spike payload/2 overwrites x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [15.0, 26.0, 26.0, 22.0, 22.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at the end of this epoch
        # is the value of x1 decayed by 4 time steps (t_epoch)
        expected_wgt_data = [10.0, 10.0, 10.0, 32.0, 32.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_bit_approx_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITH_SATURATION GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "x0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITH_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of addition
        # initial value of x1 is high to clearly see effect of saturation
        learning_dense.x1.init = np.array([115])

        # graded spike payload/2 is added to x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [115.0, 127.0, 127.0, 104.0, 104.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at pre-spike time
        # is the value of x1 decayed by 2 time steps
        expected_wgt_data = [10.0, 10.0, 10.0, 125.0, 125.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_bit_approx_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITH_SATURATION GradedSpikeCfg and
        dw=y0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "y0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITH_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of addition
        # initial value of x1 is high to clearly see effect of saturation
        learning_dense.x1.init = np.array([115])

        # graded spike payload/2 is added to x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [115.0, 127.0, 127.0, 104.0, 104.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at post-spike time
        # is the value of x1 decayed by 3 time steps
        expected_wgt_data = [10.0, 10.0, 10.0, 119.0, 119.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_sat_bit_approx_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITH_SATURATION GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        num_steps = 5

        learning_rule_cnd = "u0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITH_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(num_steps, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of overwrite
        # initial value of x1 is high to clearly see effect of saturation
        learning_dense.x1.init = np.array([115])

        # graded spike payload/2 is added to x1 trace at pre-spike time
        # value of x1 at end of the epoch is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [115.0, 127.0, 127.0, 104.0, 104.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0]
        # weight update at the end of the epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at the end of this epoch
        # is the value of x1 decayed by 4 time steps (t_epoch)
        expected_wgt_data = [10.0, 10.0, 10.0, 114.0, 114.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_bit_approx_x0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITHOUT_SATURATION GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        # Run for long enough to see two spikes
        num_steps = 10

        learning_rule_cnd = "x0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITHOUT_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(5, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of addition
        # initial value of x1 is high to clearly see effect of overflow
        learning_dense.x1.init = np.array([115])

        # graded spike payload/2 is added to x1 trace at first pre-spike time
        # x1 overflows, and only the overflow is left in x1
        # graded spike payload/2 is added to x1 trace at second pre-spike time
        # value of x1 at end of the epochs is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [115.0, 14.0, 14.0, 10.0, 10.0,
                            10.0, 36.0, 28.0, 28.0, 28.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        # for 1st spike because 1st spike caused x1 to overflow
        # x2 doesn't update at the end of the epoch for 2nd spike
        # because 2nd spike didn't cause x1 to overflow
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0,
                            16.0, 16.0, 6.0, 6.0, 6.0]
        # weight update at the end of 1st epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at pre-spike time
        # is the value of x1 decayed by 2 time steps
        # weight didn't update for 2nd spike because 2nd spike didn't cause x1
        # to overflow
        expected_wgt_data = [10.0, 10.0, 10.0, 22.0, 22.0,
                             22.0, 22.0, 22.0, 22.0, 22.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_bit_approx_y0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITHOUT_SATURATION GradedSpikeCfg and
        dw=y0 * x1 learning rule."""

        # Run for long enough to see two spikes
        num_steps = 10

        learning_rule_cnd = "y0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITHOUT_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(5, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of addition
        # initial value of x1 is high to clearly see effect of overflow
        learning_dense.x1.init = np.array([115])

        # graded spike payload/2 is added to x1 trace at first pre-spike time
        # x1 overflows, and only the overflow is left in x1
        # graded spike payload/2 is added to x1 trace at second pre-spike time
        # value of x1 at end of the epochs is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [115.0, 14.0, 14.0, 10.0, 10.0,
                            10.0, 36.0, 28.0, 28.0, 28.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        # for 1st spike because 1st spike caused x1 to overflow
        # x2 doesn't update at the end of the epoch for 2nd spike
        # because 2nd spike didn't cause x1 to overflow
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0,
                            16.0, 16.0, 6.0, 6.0, 6.0]
        # weight update at the end of 1st epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at post-spike time
        # is the value of x1 decayed by 3 time steps
        # weight didn't update for 2nd spike because 2nd spike didn't cause x1
        # to overflow
        expected_wgt_data = [10.0, 10.0, 10.0, 21.0, 21.0,
                             21.0, 21.0, 49.0, 49.0, 49.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_learning_graded_spike_add_no_sat_bit_approx_u0_condition(self):
        """Known value test for x1, x2, and weights of LearningDense with
        pre-synaptic graded spikes, ADD_WITHOUT_SATURATION GradedSpikeCfg and
        dw=x0 * x1 learning rule."""

        # Run for long enough to see two spikes
        num_steps = 10

        learning_rule_cnd = "u0"
        graded_spike_cfg = GradedSpikeCfg.ADD_WITHOUT_SATURATION

        pre_ring_buffer, learning_dense, _ = \
            self.create_network(5, learning_rule_cnd, graded_spike_cfg)

        # initialize x1 to a non-zero value to see clearly effect of addition
        # initial value of x1 is high to clearly see effect of overflow
        learning_dense.x1.init = np.array([115])

        # graded spike payload/2 is added to x1 trace at first pre-spike time
        # x1 overflows, and only the overflow is left in x1
        # graded spike payload/2 is added to x1 trace at second pre-spike time
        # value of x1 at end of the epochs is value of x1 after payload/2
        # addition decayed by 4 time steps (t_epoch)
        expected_x1_data = [115.0, 14.0, 14.0, 10.0, 10.0,
                            10.0, 36.0, 28.0, 28.0, 28.0]
        # x2 updates with decayed regular impulse, at the end of the epoch
        # for 1st spike because 1st spike caused x1 to overflow
        # x2 doesn't update at the end of the epoch for 2nd spike
        # because 2nd spike didn't cause x1 to overflow
        expected_x2_data = [0.0, 0.0, 0.0, 16.0, 16.0,
                            16.0, 16.0, 6.0, 6.0, 6.0]
        # weight update at the end of 1st epoch = value of x1 at pre-spike time
        # since at weight update, the value stored in x1 is seen as
        # x1 at the end of the previous epoch, x1 at post-spike time
        # is the value of x1 decayed by 3 time steps
        # weight didn't update for 2nd spike because 2nd spike didn't cause x1
        # to overflow
        expected_wgt_data = [10.0, 10.0, 10.0, 20.0, 20.0,
                             20.0, 20.0, 48.0, 48.0, 48.0]
        x1_data = []
        x2_data = []
        wgt_data = []
        for _ in range(num_steps):
            pre_ring_buffer.run(condition=self._run_cnd,
                                run_cfg=self._run_cfg)

            x1_data.append(learning_dense.x1.get()[0])
            x2_data.append(learning_dense.x2.get()[0])
            wgt_data.append(learning_dense.weights.get()[0, 0])

        pre_ring_buffer.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)
