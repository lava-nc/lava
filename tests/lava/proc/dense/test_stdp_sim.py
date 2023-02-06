# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.magma.core.learning.constants import GradedSpikeCfg
from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
import typing as ty
from lava.proc.lif.process import LIF, AbstractLIF, LogConfig, LearningLIF
from lava.proc.dense.process import LearningDense, Dense
from lava.proc.monitor.process import Monitor
from lava.proc.io.source import RingBuffer, PySendModelFixed, PySendModelFloat
from lava.proc.learning_rules.r_stdp_learning_rule import RewardModulatedSTDP
from lava.magma.core.model.py.neuron import (
    LearningNeuronModelFloat,
    LearningNeuronModelFixed,
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.lif.models import (
    AbstractPyLifModelFloat,
    AbstractPyLifModelFixed,
)
from lava.proc.io.source import RingBuffer as SpikeIn


class RSTDPLIF(LearningLIF):
    pass


@implements(proc=RSTDPLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class RSTDPLIFModelFloat(LearningNeuronModelFloat, AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural
    process in floating point precision with learning enabled
    to do R-STDP.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.s_out_buff = np.zeros(proc_params["shape"])

    def spiking_activation(self):
        """Spiking activation function for Learning LIF."""
        return self.v > self.vth

    def calculate_third_factor_trace(self, s_graded_in: float) -> float:
        """Generate's a third factor reward traces based on
        graded input spikes to the Learning LIF process.

        Currently, the third factor resembles the input graded spike.
        """
        return s_graded_in

    def compute_post_synaptic_trace(self, s_out_buff):
        """Compute post-synaptic trace values for this time step.

        Parameters
        ----------
        s_out_buff : ndarray
            Spikes array.

        Returns
        ----------
        result : ndarray
            Computed post synaptic trace values.
        """
        y1_tau = self._learning_rule.post_trace_decay_tau
        y1_impulse = self._learning_rule.post_trace_kernel_magnitude

        return self.y1 * np.exp(-1 / y1_tau) + y1_impulse * s_out_buff

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        s_out_y1: sends the post-synaptic spike times.
        s_out_y2: sends the graded third-factor reward signal.
        """

        self.y1 = self.compute_post_synaptic_trace(self.s_out_buff)

        super().run_spk()

        a_graded_in = self.a_third_factor_in.recv()

        self.y2 = self.calculate_third_factor_trace(a_graded_in)

        self.s_out_bap.send(self.s_out_buff)
        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)


@implements(proc=RSTDPLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("bit_accurate_loihi", "fixed_pt")
class RSTDPLIFBitAcc(LearningNeuronModelFixed, AbstractPyLifModelFixed):
    """Implementation of RSTDP Leaky-Integrate-and-Fire neural
    process bit-accurate with Loihi's hardware LIF dynamics,
    which means, it mimics Loihi behaviour bit-by-bit.

    Currently missing features (compared to Loihi 1 hardware):

    - refractory period after spiking
    - axonal delays

    Precisions of state variables

    - du: unsigned 12-bit integer (0 to 4095)
    - dv: unsigned 12-bit integer (0 to 4095)
    - bias_mant: signed 13-bit integer (-4096 to 4095). Mantissa part of neuron
      bias.
    - bias_exp: unsigned 3-bit integer (0 to 7). Exponent part of neuron bias.
    - vth: unsigned 17-bit integer (0 to 131071).

    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.effective_vth = 0
        self.s_out_buff = np.zeros(proc_params["shape"])

    def scale_threshold(self):
        """Scale threshold according to the way Loihi hardware scales it. In
        Loihi hardware, threshold is left-shifted by 6-bits to MSB-align it
        with other state variables of higher precision.
        """
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)
        self.isthrscaled = True

    def spiking_activation(self):
        """Spike when voltage exceeds threshold."""
        return self.v > self.effective_vth

    def calculate_third_factor_trace(self, s_graded_in: float) -> float:
        """Generate's a third factor reward traces based on
        graded input spikes to the Learning LIF process.

        Currently, the third factor resembles the input graded spike.
        """
        return s_graded_in

    def compute_post_synaptic_trace(self, s_out_buff):
        """Compute post-synaptic trace values for this time step.

        Parameters
        ----------
        s_out_buff : ndarray
            Spikes array.

        Returns
        ----------
        result : ndarray
            Computed post synaptic trace values.
        """
        y1_tau = self._learning_rule.post_trace_decay_tau
        y1_impulse = self._learning_rule.post_trace_kernel_magnitude

        return np.floor(self.y1 * np.exp(-1 / y1_tau) + y1_impulse * s_out_buff)

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        s_out_y1: sends the post-synaptic spike times.
        s_out_y2: sends the graded third-factor reward signal.
        """

        self.y1 = self.compute_post_synaptic_trace(self.s_out_buff)

        super().run_spk()

        a_graded_in = self.a_third_factor_in.recv()

        self.y2 = self.calculate_third_factor_trace(a_graded_in)

        self.s_out_bap.send(self.s_out_buff)
        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)


class TestSTDPSimFloatingPoint(unittest.TestCase):
    def test_stdp_floating_point(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=1,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=1,
        )

        size = 1

        weights_init = np.eye(size) * 0

        lif_0 = LIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.1)

        dense = LearningDense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.15)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(dense.s_in_bap)

        num_steps = 100

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(
            weight_after_run, np.array([[-79.35744962]])
        )

    def test_stdp_floating_point_multi_synapse(self):
        """Known value test. Run a simple learning dense layer between two LIF
        population with multiple neurons and compare to the resulting weight
        from previous runs."""
        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=1,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=1,
        )

        num_pre_neurons = 3
        num_post_neurons = 2

        weights_init = np.zeros((num_post_neurons, num_pre_neurons))

        lif_0 = LIF(
            shape=(num_pre_neurons,),
            du=0,
            dv=0,
            vth=1,
            bias_mant=np.array([0.08, 0.1, 0.11]),
        )

        dense = LearningDense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = LIF(
            shape=(num_post_neurons,),
            du=0,
            dv=0,
            vth=1,
            bias_mant=np.array([0.12, 0.15]),
        )

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(dense.s_in_bap)

        num_steps = 100

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(
            weight_after_run,
            np.array(
                [
                    [-39.5354368, -63.4727323, -80.0561724],
                    [-22.9046844, -41.8479607, -54.5550086],
                ]
            ),
        )

    def test_stdp_learning_lif_floating_point(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=1,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=1,
        )

        size = 1

        weights_init = np.eye(size) * 0

        lif_0 = LIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.1)

        dense = LearningDense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = LearningLIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.15)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(dense.s_in_bap)

        num_steps = 100

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(
            weight_after_run, np.array([[-79.35744962]])
        )


class TestSTDPSimBitApproximate(unittest.TestCase):
    def test_stdp_bit_approximate(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
            rng_seed=0,
        )

        size = 1
        weights_init = np.eye(size) * 1

        lif_0 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=25000)

        dense = LearningDense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(dense.s_in_bap)

        num_steps = 100

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(weight_after_run, np.array([[72]]))

    def test_stdp_bit_approximate_multi_synapse(self):
        """Known value test. Run a simple learning dense layer between two LIF
        population with multiple neurons and compare to the resulting weight
        from previous runs."""
        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=1,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=1,
            rng_seed=0,
        )

        num_pre_neurons = 3
        num_post_neurons = 2

        weights_init = np.zeros((num_post_neurons, num_pre_neurons))

        lif_0 = LIF(
            shape=(num_pre_neurons,),
            du=0,
            dv=0,
            vth=10000,
            bias_mant=np.array([22000, 25000, 26000]),
        )

        dense = LearningDense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = LIF(
            shape=(num_post_neurons,),
            du=0,
            dv=0,
            vth=10000,
            bias_mant=np.array([20000, 23000]),
        )

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(dense.s_in_bap)

        num_steps = 100

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(
            weight_after_run, np.array([[32.0, 3.0, -26.0], [-66.0, 26.0, 5.0]])
        )

    def test_stdp_learning_lif_bit_approximate(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
            rng_seed=0,
        )

        size = 1
        weights_init = np.eye(size) * 1

        lif_0 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=25000)

        dense = LearningDense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = LearningLIF(shape=(size,),
                            du=0,
                            dv=0,
                            vth=10000,
                            bias_mant=20000)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(dense.s_in_bap)

        num_steps = 100

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(weight_after_run, np.array([[72]]))


class TestRSTDPSimFloatingPoint(unittest.TestCase):
    def test_rstdp_floating_point(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        learning_rule = RewardModulatedSTDP(
            learning_rate=1,
            A_plus=2,
            A_minus=-2,
            pre_trace_decay_tau=10,
            post_trace_decay_tau=10,
            pre_trace_kernel_magnitude=16,
            post_trace_kernel_magnitude=16,
            eligibility_trace_decay_tau=0.5,
            t_epoch=2,
        )
        size = 1
        weights_init = np.eye(size) * 0
        num_steps = 100

        lif_0 = LIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.1)

        dense = LearningDense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = RSTDPLIF(
            shape=(size,),
            du=0,
            dv=0,
            vth=1,
            bias_mant=0.15,
            learning_rule=learning_rule,
        )

        # reward
        reward_signal = np.zeros((size, num_steps))
        reward_signal[:, num_steps // 3: num_steps // 2] = 1

        reward = SpikeIn(data=reward_signal.astype(float))
        reward_conn = Dense(weights=np.eye(size))
        reward.s_out.connect(reward_conn.s_in)
        reward_conn.a_out.connect(lif_1.a_third_factor_in)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)

        # Connect traces from LIF to Dense
        # bap: back-propagating action potential
        # y1: post-synaptic trace
        # y2: reward
        lif_1.s_out_bap.connect(dense.s_in_bap)

        lif_1.s_out_y1.connect(dense.s_in_y1)
        lif_1.s_out_y2.connect(dense.s_in_y2)
        lif_1.s_out_y3.connect(dense.s_in_y3)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(
            weight_after_run, np.array([[33.4178762]])
        )

    def test_rstdp_floating_point_multi_synapse(self):
        """Known value test. Run a simple learning dense layer between two LIF
        population with multiple neurons and compare to the resulting weight
        from previous runs."""
        learning_rule = RewardModulatedSTDP(
            learning_rate=1,
            A_plus=2,
            A_minus=-2,
            pre_trace_decay_tau=10,
            post_trace_decay_tau=10,
            pre_trace_kernel_magnitude=16,
            post_trace_kernel_magnitude=16,
            eligibility_trace_decay_tau=0.5,
            t_epoch=2,
        )

        num_pre_neurons = 3
        num_post_neurons = 2
        num_steps = 100

        weights_init = np.zeros((num_post_neurons, num_pre_neurons))

        lif_0 = LIF(
            shape=(num_pre_neurons,),
            du=0,
            dv=0,
            vth=1,
            bias_mant=np.array([0.08, 0.1, 0.11]),
        )

        dense = LearningDense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = RSTDPLIF(
            shape=(num_post_neurons,),
            du=0,
            dv=0,
            vth=1,
            bias_mant=np.array([0.12, 0.15]),
            learning_rule=learning_rule,
        )

        # reward
        reward_signal = np.zeros((num_post_neurons, num_steps))
        reward_signal[:, num_steps // 3: num_steps // 2] = 1

        reward = SpikeIn(data=reward_signal.astype(float))
        reward_conn = Dense(weights=np.eye(num_post_neurons))
        reward.s_out.connect(reward_conn.s_in)
        reward_conn.a_out.connect(lif_1.a_third_factor_in)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)

        # Connect traces from LIF to Dense
        # bap: back-propagating action potential
        # y1: post-synaptic trace
        # y2: reward
        lif_1.s_out_bap.connect(dense.s_in_bap)

        lif_1.s_out_y1.connect(dense.s_in_y1)
        lif_1.s_out_y2.connect(dense.s_in_y2)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(
            weight_after_run,
            np.array(
                [
                    [191.7346893, 31.3543832, 255.5798239],
                    [187.6966191, 17.4426083, 250.7489829],
                ]
            ),
        )


class TestRSTDPSimBitApproximate(unittest.TestCase):
    def test_rstdp_bit_approximate(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        learning_rule = RewardModulatedSTDP(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            pre_trace_decay_tau=10,
            post_trace_decay_tau=10,
            pre_trace_kernel_magnitude=16,
            post_trace_kernel_magnitude=16,
            eligibility_trace_decay_tau=0.5,
            t_epoch=1,
            rng_seed=0,
        )

        size = 1
        weights_init = np.eye(size) * 1
        num_steps = 100

        lif_0 = LIF(shape=(size,), du=0, dv=0, vth=100, bias_mant=3600)

        dense = LearningDense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = RSTDPLIF(
            shape=(size,),
            du=0,
            dv=0,
            vth=100,
            bias_mant=3700,
            learning_rule=learning_rule,
        )

        # reward
        reward_signal = np.zeros((size, num_steps))
        reward_signal[:, num_steps // 3: num_steps // 2] = 250

        reward = SpikeIn(data=reward_signal)
        reward_conn = Dense(weights=np.eye(size))
        reward.s_out.connect(reward_conn.s_in)
        reward_conn.a_out.connect(lif_1.a_third_factor_in)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)

        # Connect traces from LIF to Dense
        # bap: back-propagating action potential
        # y1: post-synaptic trace
        # y2: reward
        lif_1.s_out_bap.connect(dense.s_in_bap)

        lif_1.s_out_y1.connect(dense.s_in_y1)
        lif_1.s_out_y2.connect(dense.s_in_y2)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
        run_cnd = RunSteps(num_steps=num_steps)

        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(weight_after_run, np.array([[64]]))

    def test_rstdp_bit_approximate_multi_synapse(self):
        """Known value test. Run a simple learning dense layer between two LIF
        population with multiple neurons and compare to the resulting weight
        from previous runs."""

        learning_rule = RewardModulatedSTDP(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            pre_trace_decay_tau=10,
            post_trace_decay_tau=10,
            pre_trace_kernel_magnitude=20,
            post_trace_kernel_magnitude=20,
            eligibility_trace_decay_tau=2.4,
            t_epoch=1,
            rng_seed=0,
        )

        num_pre_neurons = 3
        num_post_neurons = 2
        num_steps = 100

        weights_init = np.zeros((num_post_neurons, num_pre_neurons))

        lif_0 = LIF(
            shape=(num_pre_neurons,),
            du=0,
            dv=0,
            vth=90,
            bias_mant=np.array([1900, 2500, 1200]),
        )

        dense = LearningDense(weights=weights_init, learning_rule=learning_rule)

        lif_1 = RSTDPLIF(
            shape=(num_post_neurons,),
            du=0,
            dv=0,
            vth=90,
            bias_mant=np.array([2400, 1600]),
            learning_rule=learning_rule,
        )

        # reward
        reward_signal = np.zeros((num_post_neurons, num_steps))
        reward_signal[:, num_steps // 3: num_steps // 2] = 16

        reward = SpikeIn(data=reward_signal.astype(float))
        reward_conn = Dense(weights=np.eye(num_post_neurons))
        reward.s_out.connect(reward_conn.s_in)
        reward_conn.a_out.connect(lif_1.a_third_factor_in)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)

        # Connect traces from LIF to Dense
        # bap: back-propagating action potential
        # y1: post-synaptic trace
        # y2: reward
        lif_1.s_out_bap.connect(dense.s_in_bap)

        lif_1.s_out_y1.connect(dense.s_in_y1)
        lif_1.s_out_y2.connect(dense.s_in_y2)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(
            weight_after_run, np.array([[3.0, 2.0, -7.0], [14.0, 19.0, 3.0]])
        )


class TestSTDPSimGradedSpikeFloatingPoint(unittest.TestCase):
    def test_stdp_graded_spike_default_mode_floating_point_x0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.8530915,
                            11.8530915, 11.8530915, 11.8530915, 7.9453649,
                            7.9453649, 7.9453649, 7.9453649, 5.3259373,
                            5.3259373, 5.3259373, 5.3259373, 3.5700826,
                            3.5700826, 3.5700826, 3.5700826, 15.49279,
                            15.49279, 15.49279, 15.49279, 10.3851277,
                            10.3851277, 10.3851277, 10.3851277, 6.9613593,
                            6.9613593, 6.9613593, 6.9613593, 4.6663387,
                            4.6663387, 4.6663387, 4.6663387, 17.605339,
                            17.605339, 17.605339, 17.605339, 11.8012117,
                            11.8012117, 11.8012117, 11.8012117, 7.9105888,
                            7.9105888, 7.9105888, 7.9105888, 5.3026262,
                            5.3026262, 5.3026262, 5.3026262, 19.5544566,
                            19.5544566, 19.5544566, 19.5544566, 13.1077443,
                            13.1077443, 13.1077443, 13.1077443]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 26.0,
                             26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0,
                             26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0,
                             44.9229364, 44.9229364, 44.9229364, 44.9229364,
                             44.9229364, 44.9229364, 44.9229364, 44.9229364,
                             44.9229364, 44.9229364, 44.9229364, 44.9229364,
                             44.9229364, 44.9229364, 44.9229364, 44.9229364,
                             64.3798451, 64.3798451, 64.3798451, 64.3798451,
                             64.3798451, 64.3798451, 64.3798451, 64.3798451,
                             64.3798451, 64.3798451, 64.3798451, 64.3798451,
                             64.3798451, 64.3798451, 64.3798451, 64.3798451,
                             83.9343017, 83.9343017, 83.9343017, 83.9343017,
                             83.9343017, 83.9343017, 83.9343017, 83.9343017]

        loihi_x1_data = [0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 7, 7, 7, 7, 4, 4,
                         4, 4, 3, 3, 3, 3, 14, 14, 14, 14, 10, 10, 10, 10, 7,
                         7, 7, 7, 5, 5, 5, 5, 17, 17, 17, 17, 11, 11, 11, 11,
                         7, 7, 7, 7, 5, 5, 5, 5, 20, 20, 20, 20, 14, 14, 14, 14]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2,
                         2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4,
                         4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3, 3, 3, 3,
                         2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 26, 26, 26, 26, 26, 26,
                          26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 44, 44, 44,
                          44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
                          64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                          64, 64, 64, 84, 84, 84, 84, 84, 84, 84, 84]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="x0 * x1",
                                            x1_impulse=16, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.DEFAULT)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        # plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        # plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        # plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_default_mode_floating_point_y0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.8530915,
                            11.8530915, 11.8530915, 11.8530915, 7.9453649,
                            7.9453649, 7.9453649, 7.9453649, 5.3259373,
                            5.3259373, 5.3259373, 5.3259373, 3.5700826,
                            3.5700826, 3.5700826, 3.5700826, 15.49279,
                            15.49279, 15.49279, 15.49279, 10.3851277,
                            10.3851277, 10.3851277, 10.3851277, 6.9613593,
                            6.9613593, 6.9613593, 6.9613593, 4.6663387,
                            4.6663387, 4.6663387, 4.6663387, 17.605339,
                            17.605339, 17.605339, 17.605339, 11.8012117,
                            11.8012117, 11.8012117, 11.8012117, 7.9105888,
                            7.9105888, 7.9105888, 7.9105888, 5.3026262,
                            5.3026262, 5.3026262, 5.3026262, 19.5544566,
                            19.5544566, 19.5544566, 19.5544566, 13.1077443,
                            13.1077443, 13.1077443, 13.1077443]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             28.9229364, 28.9229364, 28.9229364, 28.9229364,
                             28.9229364, 28.9229364, 28.9229364, 28.9229364,
                             28.9229364, 28.9229364, 28.9229364, 28.9229364,
                             28.9229364, 28.9229364, 28.9229364, 28.9229364,
                             46.5282754, 46.5282754, 46.5282754, 46.5282754,
                             46.5282754, 46.5282754, 46.5282754, 46.5282754,
                             46.5282754, 46.5282754, 46.5282754, 46.5282754,
                             46.5282754, 46.5282754, 46.5282754, 46.5282754,
                             46.5282754, 46.5282754, 46.5282754, 46.5282754,
                             62.5381104, 62.5381104, 62.5381104, 62.5381104]

        loihi_x1_data = [0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 7, 7, 7, 7, 4, 4,
                         4, 4, 3, 3, 3, 3, 14, 14, 14, 14, 10, 10, 10, 10, 7,
                         7, 7, 7, 5, 5, 5, 5, 17, 17, 17, 17, 11, 11, 11, 11,
                         7, 7, 7, 7, 5, 5, 5, 5, 20, 20, 20, 20, 14, 14, 14, 14]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2,
                         2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4,
                         4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3, 3, 3, 3,
                         2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                          10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 28, 28, 28,
                          28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
                          45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                          45, 45, 45, 45, 45, 45, 45, 61, 61, 61, 61]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="y0 * x1",
                                            x1_impulse=16, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo
        # o o o ! o o o o o ! oo oo oo oo oo !! oo oo oo oo oo !! oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.DEFAULT)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_default_mode_floating_point_u0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.8530915,
                            11.8530915, 11.8530915, 11.8530915, 7.9453649,
                            7.9453649, 7.9453649, 7.9453649, 5.3259373,
                            5.3259373, 5.3259373, 5.3259373, 3.5700826,
                            3.5700826, 3.5700826, 3.5700826, 15.49279,
                            15.49279, 15.49279, 15.49279, 10.3851277,
                            10.3851277, 10.3851277, 10.3851277, 6.9613593,
                            6.9613593, 6.9613593, 6.9613593, 4.6663387,
                            4.6663387, 4.6663387, 4.6663387, 17.605339,
                            17.605339, 17.605339, 17.605339, 11.8012117,
                            11.8012117, 11.8012117, 11.8012117, 7.9105888,
                            7.9105888, 7.9105888, 7.9105888, 5.3026262,
                            5.3026262, 5.3026262, 5.3026262, 19.5544566,
                            19.5544566, 19.5544566, 19.5544566, 13.1077443,
                            13.1077443, 13.1077443, 13.1077443]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             21.8530915, 21.8530915, 21.8530915, 21.8530915,
                             29.7984564, 29.7984564, 29.7984564, 29.7984564,
                             35.1243937, 35.1243937, 35.1243937, 35.1243937,
                             38.6944763, 38.6944763, 38.6944763, 38.6944763,
                             54.1872662, 54.1872662, 54.1872662, 54.1872662,
                             64.5723939, 64.5723939, 64.5723939, 64.5723939,
                             71.5337532, 71.5337532, 71.5337532, 71.5337532,
                             76.2000919, 76.2000919, 76.2000919, 76.2000919,
                             93.8054309, 93.8054309, 93.8054309, 93.8054309,
                             105.6066426, 105.6066426, 105.6066426,
                             105.6066426, 113.5172313, 113.5172313,
                             113.5172313, 113.5172313, 118.8198575,
                             118.8198575, 118.8198575, 118.8198575,
                             138.3743142, 138.3743142, 138.3743142,
                             138.3743142, 151.4820585, 151.4820585,
                             151.4820585, 151.4820585]

        loihi_x1_data = [0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 7, 7, 7, 7, 4, 4,
                         4, 4, 3, 3, 3, 3, 14, 14, 14, 14, 10, 10, 10, 10, 7,
                         7, 7, 7, 5, 5, 5, 5, 17, 17, 17, 17, 11, 11, 11, 11,
                         7, 7, 7, 7, 5, 5, 5, 5, 20, 20, 20, 20, 14, 14, 14, 14]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2,
                         2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4,
                         4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3, 3, 3, 3,
                         2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 21, 21, 21, 21, 28, 28,
                          28, 28, 32, 32, 32, 32, 35, 35, 35, 35, 49, 49, 49,
                          49, 59, 59, 59, 59, 66, 66, 66, 66, 71, 71, 71, 71,
                          88, 88, 88, 88, 99, 99, 99, 99, 106, 106, 106, 106,
                          111, 111, 111, 111, 131, 131, 131, 131, 145, 145,
                          145, 145]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="u0 * x1",
                                            x1_impulse=16, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo
        # o o o ! o o o o o ! oo oo oo oo oo !! oo oo oo oo oo !! oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.DEFAULT)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_overwrite_mode_floating_point_x0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 16.7580012,
                            16.7580012, 16.7580012, 16.7580012, 11.2332241,
                            11.2332241, 11.2332241, 11.2332241, 7.5298553,
                            7.5298553, 7.5298553, 7.5298553, 5.0474129,
                            5.0474129, 23.0, 23.0, 15.4173611, 15.4173611,
                            15.4173611, 15.4173611, 10.3345662, 10.3345662,
                            10.3345662, 10.3345662, 6.9274669, 6.9274669,
                            6.9274669, 6.9274669, 4.6436199, 4.6436199,
                            4.6436199, 17.0, 11.3954408, 11.3954408,
                            11.3954408, 11.3954408, 7.6385924, 7.6385924,
                            7.6385924, 7.6385924, 5.1203016, 5.1203016,
                            5.1203016, 5.1203016, 3.4322408, 3.4322408,
                            3.4322408, 3.4322408, 9.0493206, 9.0493206,
                            9.0493206, 9.0493206, 6.065941, 6.065941, 6.065941,
                            6.065941]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             51.4517428, 51.4517428, 51.4517428, 51.4517428,
                             51.4517428, 51.4517428, 51.4517428, 51.4517428,
                             51.4517428, 51.4517428, 51.4517428, 51.4517428,
                             51.4517428, 51.4517428, 51.4517428, 51.4517428,
                             64.0456525, 64.0456525, 64.0456525, 64.0456525,
                             64.0456525, 64.0456525, 64.0456525, 64.0456525,
                             64.0456525, 64.0456525, 64.0456525, 64.0456525,
                             64.0456525, 64.0456525, 64.0456525, 64.0456525,
                             73.0949731, 73.0949731, 73.0949731, 73.0949731,
                             73.0949731, 73.0949731, 73.0949731, 73.0949731]

        loihi_x1_data = [0, 0, 0, 0, 25, 25, 25, 16, 16, 16, 16, 10, 10, 10,
                         10, 6, 6, 6, 6, 4, 4, 23, 23, 14, 14, 14, 14, 10, 10,
                         10, 10, 7, 7, 7, 7, 5, 5, 5, 17, 10, 10, 10, 10, 6, 6,
                         6, 6, 3, 3, 3, 3, 2, 2, 2, 2, 10, 10, 10, 10, 7, 7, 7,
                         7]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2,
                         2, 2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4,
                         4, 4, 4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3,
                         3, 3, 3, 2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 32, 32, 32, 32, 32, 32,
                          32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 50, 50, 50,
                          50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                          62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62,
                          62, 62, 62, 72, 72, 72, 72, 72, 72, 72, 72]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="x0 * x1",
                                            x1_impulse=0, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.OVERWRITE)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_overwrite_mode_floating_point_y0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 16.7580012,
                            16.7580012, 16.7580012, 16.7580012, 11.2332241,
                            11.2332241, 11.2332241, 11.2332241, 7.5298553,
                            7.5298553, 7.5298553, 7.5298553, 5.0474129,
                            5.0474129, 23.0, 23.0, 15.4173611, 15.4173611,
                            15.4173611, 15.4173611, 10.3345662, 10.3345662,
                            10.3345662, 10.3345662, 6.9274669, 6.9274669,
                            6.9274669, 6.9274669, 4.6436199, 4.6436199,
                            4.6436199, 17.0, 11.3954408, 11.3954408,
                            11.3954408, 11.3954408, 7.6385924, 7.6385924,
                            7.6385924, 7.6385924, 5.1203016, 5.1203016,
                            5.1203016, 5.1203016, 3.4322408, 3.4322408,
                            3.4322408, 3.4322408, 9.0493206, 9.0493206,
                            9.0493206, 9.0493206, 6.065941, 6.065941, 6.065941,
                            6.065941]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             28.8308073, 28.8308073, 28.8308073, 28.8308073,
                             28.8308073, 28.8308073, 28.8308073, 28.8308073,
                             28.8308073, 28.8308073, 28.8308073, 28.8308073,
                             28.8308073, 28.8308073, 28.8308073, 28.8308073,
                             40.2262481, 40.2262481, 40.2262481, 40.2262481,
                             40.2262481, 40.2262481, 40.2262481, 40.2262481,
                             40.2262481, 40.2262481, 40.2262481, 40.2262481,
                             40.2262481, 40.2262481, 40.2262481, 40.2262481,
                             40.2262481, 40.2262481, 40.2262481, 40.2262481,
                             47.6352052, 47.6352052, 47.6352052, 47.6352052]

        loihi_x1_data = [0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 7, 7, 7, 7, 4, 4,
                         4, 4, 3, 3, 3, 3, 14, 14, 14, 14, 10, 10, 10, 10, 7,
                         7, 7, 7, 5, 5, 5, 5, 17, 17, 17, 17, 11, 11, 11, 11,
                         7, 7, 7, 7, 5, 5, 5, 5, 20, 20, 20, 20, 14, 14, 14, 14]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2,
                         2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4,
                         4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3, 3, 3, 3,
                         2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                          10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 28, 28, 28,
                          28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
                          38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38,
                          38, 38, 38, 38, 38, 38, 38, 46, 46, 46, 46]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="y0 * x1",
                                            x1_impulse=0, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo
        # o o o ! o o o o o ! oo oo oo oo oo !! oo oo oo oo oo !! oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.OVERWRITE)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_overwrite_mode_floating_point_u0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 16.7580012,
                            16.7580012, 16.7580012, 16.7580012, 11.2332241,
                            11.2332241, 11.2332241, 11.2332241, 7.5298553,
                            7.5298553, 7.5298553, 7.5298553, 5.0474129,
                            5.0474129, 23.0, 23.0, 15.4173611, 15.4173611,
                            15.4173611, 15.4173611, 10.3345662, 10.3345662,
                            10.3345662, 10.3345662, 6.9274669, 6.9274669,
                            6.9274669, 6.9274669, 4.6436199, 4.6436199,
                            4.6436199, 17.0, 11.3954408, 11.3954408,
                            11.3954408, 11.3954408, 7.6385924, 7.6385924,
                            7.6385924, 7.6385924, 5.1203016, 5.1203016,
                            5.1203016, 5.1203016, 3.4322408, 3.4322408,
                            3.4322408, 3.4322408, 9.0493206, 9.0493206,
                            9.0493206, 9.0493206, 6.065941, 6.065941, 6.065941,
                            6.065941]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             26.7580012, 26.7580012, 26.7580012, 26.7580012,
                             37.9912253, 37.9912253, 37.9912253, 37.9912253,
                             45.5210806, 45.5210806, 45.5210806, 45.5210806,
                             50.5684935, 50.5684935, 50.5684935, 50.5684935,
                             65.9858546, 65.9858546, 65.9858546, 65.9858546,
                             76.3204207, 76.3204207, 76.3204207, 76.3204207,
                             83.2478876, 83.2478876, 83.2478876, 83.2478876,
                             87.8915075, 87.8915075, 87.8915075, 87.8915075,
                             99.2869483, 99.2869483, 99.2869483, 99.2869483,
                             106.9255407, 106.9255407, 106.9255407,
                             106.9255407, 112.0458423, 112.0458423,
                             112.0458423, 112.0458423, 115.4780831,
                             115.4780831, 115.4780831, 115.4780831,
                             124.5274037, 124.5274037, 124.5274037,
                             124.5274037, 130.5933447, 130.5933447,
                             130.5933447, 130.5933447]

        loihi_x1_data = [0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 7, 7, 7, 7, 4, 4,
                         4, 4, 3, 3, 3, 3, 14, 14, 14, 14, 10, 10, 10, 10, 7,
                         7, 7, 7, 5, 5, 5, 5, 17, 17, 17, 17, 11, 11, 11, 11,
                         7, 7, 7, 7, 5, 5, 5, 5, 20, 20, 20, 20, 14, 14, 14, 14]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2,
                         2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4,
                         4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3, 3, 3, 3,
                         2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 26, 26, 26, 26, 36, 36,
                          36, 36, 42, 42, 42, 42, 46, 46, 46, 46, 60, 60, 60,
                          60, 70, 70, 70, 70, 77, 77, 77, 77, 82, 82, 82, 82,
                          92, 92, 92, 92, 98, 98, 98, 98, 101, 101, 101, 101,
                          103, 103, 103, 103, 113, 113, 113, 113, 120, 120,
                          120, 120]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="u0 * x1",
                                            x1_impulse=0, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo
        # o o o ! o o o o o ! oo oo oo oo oo !! oo oo oo oo oo !! oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.OVERWRITE)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_add_saturation_mode_floating_point_x0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 16.7580012,
                            16.7580012, 16.7580012, 16.7580012, 11.2332241,
                            11.2332241, 11.2332241, 11.2332241, 7.5298553,
                            7.5298553, 7.5298553, 7.5298553, 5.0474129,
                            5.0474129, 28.0474129, 28.0474129, 18.8007431,
                            18.8007431, 18.8007431, 18.8007431, 12.602515,
                            12.602515, 12.602515, 12.602515, 8.4477184,
                            8.4477184, 8.4477184, 8.4477184, 5.662675,
                            5.662675, 5.662675, 22.662675, 15.1912454,
                            15.1912454, 15.1912454, 15.1912454, 10.1829963,
                            10.1829963, 10.1829963, 10.1829963, 6.8258665,
                            6.8258665, 6.8258665, 6.8258665, 4.5755152,
                            4.5755152, 4.5755152, 4.5755152, 12.1163802,
                            12.1163802, 12.1163802, 12.1163802, 8.1218525,
                            8.1218525, 8.1218525, 8.1218525]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             55.584215, 55.584215, 55.584215, 55.584215,
                             55.584215, 55.584215, 55.584215, 55.584215,
                             55.584215, 55.584215, 55.584215, 55.584215,
                             55.584215, 55.584215, 55.584215, 55.584215,
                             72.3731376, 72.3731376, 72.3731376, 72.3731376,
                             72.3731376, 72.3731376, 72.3731376, 72.3731376,
                             72.3731376, 72.3731376, 72.3731376, 72.3731376,
                             72.3731376, 72.3731376, 72.3731376, 72.3731376,
                             84.4895177, 84.4895177, 84.4895177, 84.4895177,
                             84.4895177, 84.4895177, 84.4895177, 84.4895177]

        loihi_x1_data = [0, 0, 0, 0, 25, 25, 25, 16, 16, 16, 16, 10, 10, 10,
                         10, 6, 6, 6, 6, 4, 4, 27, 27, 17, 17, 17, 17, 12, 12,
                         12, 12, 8, 8, 8, 8, 6, 6, 6, 23, 15, 15, 15, 15, 9,
                         9, 9, 9, 5, 5, 5, 5, 4, 4, 4, 4, 12, 12, 12, 12, 8, 8,
                         8, 8]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2,
                         2, 2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4,
                         4, 4, 4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3,
                         3, 3, 3, 2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 32, 32, 32, 32, 32, 32,
                          32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 53, 53, 53,
                          53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53,
                          70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
                          70, 70, 70, 82, 82, 82, 82, 82, 82, 82, 82]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="x0 * x1",
                                            x1_impulse=0, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 255
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.ADD_SATURATION)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_add_saturation_mode_floating_point_y0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 16.7580012,
                            16.7580012, 16.7580012, 16.7580012, 11.2332241,
                            11.2332241, 11.2332241, 11.2332241, 7.5298553,
                            7.5298553, 7.5298553, 7.5298553, 5.0474129,
                            5.0474129, 28.0474129, 28.0474129, 18.8007431,
                            18.8007431, 18.8007431, 18.8007431, 12.602515,
                            12.602515, 12.602515, 12.602515, 8.4477184,
                            8.4477184, 8.4477184, 8.4477184, 5.662675,
                            5.662675, 5.662675, 22.662675, 15.1912454,
                            15.1912454, 15.1912454, 15.1912454, 10.1829963,
                            10.1829963, 10.1829963, 10.1829963, 6.8258665,
                            6.8258665, 6.8258665, 6.8258665, 4.5755152,
                            4.5755152, 4.5755152, 4.5755152, 12.1163802,
                            12.1163802, 12.1163802, 12.1163802, 8.1218525,
                            8.1218525, 8.1218525, 8.1218525]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             32.9632795, 32.9632795, 32.9632795, 32.9632795,
                             32.9632795, 32.9632795, 32.9632795, 32.9632795,
                             32.9632795, 32.9632795, 32.9632795, 32.9632795,
                             32.9632795, 32.9632795, 32.9632795, 32.9632795,
                             48.1545249, 48.1545249, 48.1545249, 48.1545249,
                             48.1545249, 48.1545249, 48.1545249, 48.1545249,
                             48.1545249, 48.1545249, 48.1545249, 48.1545249,
                             48.1545249, 48.1545249, 48.1545249, 48.1545249,
                             48.1545249, 48.1545249, 48.1545249, 48.1545249,
                             58.0745779, 58.0745779, 58.0745779, 58.0745779]

        loihi_x1_data = [0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 7, 7, 7, 7, 4, 4,
                         4, 4, 3, 3, 3, 3, 14, 14, 14, 14, 10, 10, 10, 10, 7,
                         7, 7, 7, 5, 5, 5, 5, 17, 17, 17, 17, 11, 11, 11, 11,
                         7, 7, 7, 7, 5, 5, 5, 5, 20, 20, 20, 20, 14, 14, 14, 14]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2,
                         2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4,
                         4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3, 3, 3, 3,
                         2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                          10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 31, 31, 31,
                          31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
                          46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                          46, 46, 46, 46, 46, 46, 46, 55, 55, 55, 55]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="y0 * x1",
                                            x1_impulse=0, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo
        # o o o ! o o o o o ! oo oo oo oo oo !! oo oo oo oo oo !! oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.ADD_SATURATION)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_add_saturation_mode_floating_point_u0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 16.7580012,
                            16.7580012, 16.7580012, 16.7580012, 11.2332241,
                            11.2332241, 11.2332241, 11.2332241, 7.5298553,
                            7.5298553, 7.5298553, 7.5298553, 5.0474129,
                            5.0474129, 28.0474129, 28.0474129, 18.8007431,
                            18.8007431, 18.8007431, 18.8007431, 12.602515,
                            12.602515, 12.602515, 12.602515, 8.4477184,
                            8.4477184, 8.4477184, 8.4477184, 5.662675,
                            5.662675, 5.662675, 22.662675, 15.1912454,
                            15.1912454, 15.1912454, 15.1912454, 10.1829963,
                            10.1829963, 10.1829963, 10.1829963, 6.8258665,
                            6.8258665, 6.8258665, 6.8258665, 4.5755152,
                            4.5755152, 4.5755152, 4.5755152, 12.1163802,
                            12.1163802, 12.1163802, 12.1163802, 8.1218525,
                            8.1218525, 8.1218525, 8.1218525]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             26.7580012, 26.7580012, 26.7580012, 26.7580012,
                             37.9912253, 37.9912253, 37.9912253, 37.9912253,
                             45.5210806, 45.5210806, 45.5210806, 45.5210806,
                             50.5684935, 50.5684935, 50.5684935, 50.5684935,
                             65.9858546, 65.9858546, 65.9858546, 65.9858546,
                             76.3204207, 76.3204207, 76.3204207, 76.3204207,
                             83.2478876, 83.2478876, 83.2478876, 83.2478876,
                             87.8915075, 87.8915075, 87.8915075, 87.8915075,
                             99.2869483, 99.2869483, 99.2869483, 99.2869483,
                             106.9255407, 106.9255407, 106.9255407,
                             106.9255407, 112.0458423, 112.0458423,
                             112.0458423, 112.0458423, 115.4780831,
                             115.4780831, 115.4780831, 115.4780831,
                             124.5274037, 124.5274037, 124.5274037,
                             124.5274037, 130.5933447, 130.5933447,
                             130.5933447, 130.5933447]

        loihi_x1_data = [0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 7, 7, 7, 7, 4, 4,
                         4, 4, 3, 3, 3, 3, 14, 14, 14, 14, 10, 10, 10, 10, 7,
                         7, 7, 7, 5, 5, 5, 5, 17, 17, 17, 17, 11, 11, 11, 11,
                         7, 7, 7, 7, 5, 5, 5, 5, 20, 20, 20, 20, 14, 14, 14, 14]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2,
                         2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4,
                         4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3, 3, 3, 3,
                         2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 26, 26, 26, 26, 36, 36,
                          36, 36, 42, 42, 42, 42, 46, 46, 46, 46, 63, 63, 63,
                          63, 75, 75, 75, 75, 83, 83, 83, 83, 89, 89, 89, 89,
                          104, 104, 104, 104, 113, 113, 113, 113, 118, 118,
                          118, 118, 122, 122, 122, 122, 134, 134, 134, 134,
                          142, 142, 142, 142]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="u0 * x1",
                                            x1_impulse=0, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo
        # o o o ! o o o o o ! oo oo oo oo oo !! oo oo oo oo oo !! oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.ADD_SATURATION)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_add_no_saturation_mode_floating_point_x0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 16.7580012,
                            16.7580012, 16.7580012, 16.7580012, 11.2332241,
                            11.2332241, 11.2332241, 11.2332241, 7.5298553,
                            7.5298553, 7.5298553, 7.5298553, 5.0474129,
                            5.0474129, 28.0474129, 28.0474129, 18.8007431,
                            18.8007431, 18.8007431, 18.8007431, 12.602515,
                            12.602515, 12.602515, 12.602515, 8.4477184,
                            8.4477184, 8.4477184, 8.4477184, 5.662675,
                            5.662675, 5.662675, 22.662675, 15.1912454,
                            15.1912454, 15.1912454, 15.1912454, 10.1829963,
                            10.1829963, 10.1829963, 10.1829963, 6.8258665,
                            6.8258665, 6.8258665, 6.8258665, 4.5755152,
                            4.5755152, 4.5755152, 4.5755152, 12.1163802,
                            12.1163802, 12.1163802, 12.1163802, 8.1218525,
                            8.1218525, 8.1218525, 8.1218525]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             32.6209355, 32.6209355, 32.6209355, 32.6209355,
                             55.584215, 55.584215, 55.584215, 55.584215,
                             55.584215, 55.584215, 55.584215, 55.584215,
                             55.584215, 55.584215, 55.584215, 55.584215,
                             55.584215, 55.584215, 55.584215, 55.584215,
                             72.3731376, 72.3731376, 72.3731376, 72.3731376,
                             72.3731376, 72.3731376, 72.3731376, 72.3731376,
                             72.3731376, 72.3731376, 72.3731376, 72.3731376,
                             72.3731376, 72.3731376, 72.3731376, 72.3731376,
                             84.4895177, 84.4895177, 84.4895177, 84.4895177,
                             84.4895177, 84.4895177, 84.4895177, 84.4895177]

        loihi_x1_data = [0, 0, 0, 0, 25, 25, 25, 16, 16, 16, 16, 10, 10, 10,
                         10, 6, 6, 6, 6, 4, 4, 27, 27, 17, 17, 17, 17, 12, 12,
                         12, 12, 8, 8, 8, 8, 6, 6, 6, 23, 15, 15, 15, 15, 9,
                         9, 9, 9, 5, 5, 5, 5, 4, 4, 4, 4, 12, 12, 12, 12, 8, 8,
                         8, 8]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2,
                         2, 2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4,
                         4, 4, 4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3,
                         3, 3, 3, 2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 32, 32, 32, 32, 32, 32,
                          32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 53, 53, 53,
                          53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53,
                          70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,
                          70, 70, 70, 82, 82, 82, 82, 82, 82, 82, 82]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="x0 * x1",
                                            x1_impulse=0, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.ADD_NO_SATURATION)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_add_no_saturation_mode_floating_point_y0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 16.7580012,
                            16.7580012, 16.7580012, 16.7580012, 11.2332241,
                            11.2332241, 11.2332241, 11.2332241, 7.5298553,
                            7.5298553, 7.5298553, 7.5298553, 5.0474129,
                            5.0474129, 28.0474129, 28.0474129, 18.8007431,
                            18.8007431, 18.8007431, 18.8007431, 12.602515,
                            12.602515, 12.602515, 12.602515, 8.4477184,
                            8.4477184, 8.4477184, 8.4477184, 5.662675,
                            5.662675, 5.662675, 22.662675, 15.1912454,
                            15.1912454, 15.1912454, 15.1912454, 10.1829963,
                            10.1829963, 10.1829963, 10.1829963, 6.8258665,
                            6.8258665, 6.8258665, 6.8258665, 4.5755152,
                            4.5755152, 4.5755152, 4.5755152, 12.1163802,
                            12.1163802, 12.1163802, 12.1163802, 8.1218525,
                            8.1218525, 8.1218525, 8.1218525]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             32.9632795, 32.9632795, 32.9632795, 32.9632795,
                             32.9632795, 32.9632795, 32.9632795, 32.9632795,
                             32.9632795, 32.9632795, 32.9632795, 32.9632795,
                             32.9632795, 32.9632795, 32.9632795, 32.9632795,
                             48.1545249, 48.1545249, 48.1545249, 48.1545249,
                             48.1545249, 48.1545249, 48.1545249, 48.1545249,
                             48.1545249, 48.1545249, 48.1545249, 48.1545249,
                             48.1545249, 48.1545249, 48.1545249, 48.1545249,
                             48.1545249, 48.1545249, 48.1545249, 48.1545249,
                             58.0745779, 58.0745779, 58.0745779, 58.0745779]

        loihi_x1_data = [0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 7, 7, 7, 7, 4, 4,
                         4, 4, 3, 3, 3, 3, 14, 14, 14, 14, 10, 10, 10, 10, 7,
                         7, 7, 7, 5, 5, 5, 5, 17, 17, 17, 17, 11, 11, 11, 11,
                         7, 7, 7, 7, 5, 5, 5, 5, 20, 20, 20, 20, 14, 14, 14, 14]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2,
                         2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4,
                         4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3, 3, 3, 3,
                         2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                          10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 31, 31, 31,
                          31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
                          46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                          46, 46, 46, 46, 46, 46, 46, 55, 55, 55, 55]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="y0 * x1",
                                            x1_impulse=0, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo
        # o o o ! o o o o o ! oo oo oo oo oo !! oo oo oo oo oo !! oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.ADD_NO_SATURATION)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)

    def test_stdp_graded_spike_add_no_saturation_mode_floating_point_u0_condition(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 16.7580012,
                            16.7580012, 16.7580012, 16.7580012, 11.2332241,
                            11.2332241, 11.2332241, 11.2332241, 7.5298553,
                            7.5298553, 7.5298553, 7.5298553, 5.0474129,
                            5.0474129, 28.0474129, 28.0474129, 18.8007431,
                            18.8007431, 18.8007431, 18.8007431, 12.602515,
                            12.602515, 12.602515, 12.602515, 8.4477184,
                            8.4477184, 8.4477184, 8.4477184, 5.662675,
                            5.662675, 5.662675, 22.662675, 15.1912454,
                            15.1912454, 15.1912454, 15.1912454, 10.1829963,
                            10.1829963, 10.1829963, 10.1829963, 6.8258665,
                            6.8258665, 6.8258665, 6.8258665, 4.5755152,
                            4.5755152, 4.5755152, 4.5755152, 12.1163802,
                            12.1163802, 12.1163802, 12.1163802, 8.1218525,
                            8.1218525, 8.1218525, 8.1218525]
        expected_x2_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1714793,
                            13.1714793, 13.1714793, 13.1714793, 5.9183271,
                            5.9183271, 5.9183271, 5.9183271, 2.6592758,
                            2.6592758, 2.6592758, 2.6592758, 1.1948896,
                            1.1948896, 1.1948896, 1.1948896, 16.6245796,
                            16.6245796, 16.6245796, 16.6245796, 7.4699051,
                            7.4699051, 7.4699051, 7.4699051, 3.3564447,
                            3.3564447, 3.3564447, 3.3564447, 1.5081478,
                            1.5081478, 1.5081478, 1.5081478, 20.3271926,
                            20.3271926, 20.3271926, 20.3271926, 9.1335964,
                            9.1335964, 9.1335964, 9.1335964, 4.1039894,
                            4.1039894, 4.1039894, 4.1039894, 1.8440413,
                            1.8440413, 1.8440413, 1.8440413, 24.8285812,
                            24.8285812, 24.8285812, 24.8285812, 11.1562007,
                            11.1562007, 11.1562007, 11.1562007]
        expected_wgt_data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                             26.7580012, 26.7580012, 26.7580012, 26.7580012,
                             37.9912253, 37.9912253, 37.9912253, 37.9912253,
                             45.5210806, 45.5210806, 45.5210806, 45.5210806,
                             50.5684935, 50.5684935, 50.5684935, 50.5684935,
                             65.9858546, 65.9858546, 65.9858546, 65.9858546,
                             76.3204207, 76.3204207, 76.3204207, 76.3204207,
                             83.2478876, 83.2478876, 83.2478876, 83.2478876,
                             87.8915075, 87.8915075, 87.8915075, 87.8915075,
                             99.2869483, 99.2869483, 99.2869483, 99.2869483,
                             106.9255407, 106.9255407, 106.9255407,
                             106.9255407, 112.0458423, 112.0458423,
                             112.0458423, 112.0458423, 115.4780831,
                             115.4780831, 115.4780831, 115.4780831,
                             124.5274037, 124.5274037, 124.5274037,
                             124.5274037, 130.5933447, 130.5933447,
                             130.5933447, 130.5933447]

        loihi_x1_data = [0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 7, 7, 7, 7, 4, 4,
                         4, 4, 3, 3, 3, 3, 14, 14, 14, 14, 10, 10, 10, 10, 7,
                         7, 7, 7, 5, 5, 5, 5, 17, 17, 17, 17, 11, 11, 11, 11,
                         7, 7, 7, 7, 5, 5, 5, 5, 20, 20, 20, 20, 14, 14, 14, 14]
        loihi_x2_data = [0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 5, 5, 5, 5, 2, 2,
                         2, 2, 1, 1, 1, 1, 15, 15, 15, 15, 7, 7, 7, 7, 4, 4, 4,
                         4, 2, 2, 2, 2, 20, 20, 20, 20, 8, 8, 8, 8, 3, 3, 3, 3,
                         2, 2, 2, 2, 25, 25, 25, 25, 12, 12, 12, 12]
        loihi_wgt_data = [10, 10, 10, 10, 10, 10, 10, 26, 26, 26, 26, 36, 36,
                          36, 36, 42, 42, 42, 42, 46, 46, 46, 46, 63, 63, 63,
                          63, 75, 75, 75, 75, 83, 83, 83, 83, 89, 89, 89, 89,
                          104, 104, 104, 104, 113, 113, 113, 113, 118, 118,
                          118, 118, 122, 122, 122, 122, 134, 134, 134, 134,
                          142, 142, 142, 142]

        exception_map = {
            RingBuffer: PySendModelFloat
        }

        num_steps = 63

        size = 1
        weights_init = np.eye(size) * 10

        learning_rule = Loihi2FLearningRule(dw="u0 * x1",
                                            x1_impulse=0, x1_tau=10,
                                            x2_impulse=24, x2_tau=5,
                                            y1_impulse=16, y1_tau=10,
                                            y2_impulse=24, y2_tau=5,
                                            t_epoch=4)

        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # o o o ! o o o ! o o oo !! oo oo oo !! oo oo oo !! oo oo oo !!
        # o o o o ! o o o o ! oo oo oo oo !! oo oo oo oo !! oo oo oo oo
        # o o o ! o o o o o ! oo oo oo oo oo !! oo oo oo oo oo !! oo oo

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 4] = 50
        spike_raster_pre[0, 21] = 46
        spike_raster_pre[0, 38] = 34
        spike_raster_pre[0, 55] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.ADD_NO_SATURATION)

        spike_raster_post = np.zeros((size, num_steps))
        spike_raster_post[0, 3] = 1
        spike_raster_post[0, 21] = 1
        spike_raster_post[0, 39] = 1
        spike_raster_post[0, 57] = 1
        pattern_post = RingBuffer(data=spike_raster_post.astype(int))

        pattern_pre.s_out.connect(learning_dense.s_in)
        pattern_post.s_out.connect(learning_dense.s_in_bap)

        monitor_x1 = Monitor()
        monitor_x1.probe(target=learning_dense.x1, num_steps=num_steps)
        monitor_x2 = Monitor()
        monitor_x2.probe(target=learning_dense.x2, num_steps=num_steps)
        monitor_wgt = Monitor()
        monitor_wgt.probe(target=learning_dense.weights, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor_x1.get_data()["learning_dense"]["x1"][:, 0]
        x2_data = monitor_x2.get_data()["learning_dense"]["x2"][:, 0]
        wgt_data = monitor_wgt.get_data()["learning_dense"]["weights"][:, 0, 0]

        pattern_pre.stop()

        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        list_x2_data = [round(data_pt, 7) for data_pt in x2_data.tolist()]
        list_wgt_data = [round(data_pt, 7) for data_pt in wgt_data.tolist()]

        print("list_x1_data")
        print(list_x1_data)
        print("list_x2_data")
        print(list_x2_data)
        print("list_wgt_data")
        print(list_wgt_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.title("x1")
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x1_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("x2")
        plt.step(list(range(num_steps)), x2_data, label="data")
        plt.step(list(range(num_steps)), expected_x2_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_x2_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.title("wgt")
        plt.step(list(range(num_steps)), wgt_data, label="data")
        plt.step(list(range(num_steps)), expected_wgt_data, label="expected_data")
        plt.step(list(range(num_steps)), loihi_wgt_data,
                 label="loihi_data")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
        np.testing.assert_almost_equal(x2_data, expected_x2_data)
        np.testing.assert_almost_equal(wgt_data, expected_wgt_data)


class TestSTDPSimGradedSpikeBitApproximate(unittest.TestCase):
    def test_stdp_graded_spike_default_mode_bit_approximate(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 14.0, 14.0, 12.0, 12.0, 10.0, 10.0,
                            8.0, 8.0, 6.0, 6.0, 4.0, 4.0, 17.0, 17.0, 15.0,
                            15.0, 13.0, 13.0, 11.0, 11.0, 9.0, 9.0, 7.0, 7.0,
                            5.0, 5.0, 18.0, 18.0, 16.0, 16.0, 29.0, 29.0, 24.0,
                            24.0, 19.0, 19.0, 15.0, 15.0, 12.0, 12.0, 10.0,
                            10.0, 8.0, 8.0, 8.0, 8.0, 22.0, 22.0, 18.0, 18.0,
                            15.0, 15.0, 13.0, 13.0, 11.0, 11.0, 9.0, 9.0, 9.0,
                            9.0, 7.0, 7.0, 21.0, 21.0, 17.0, 17.0, 13.0, 13.0,
                            11.0, 11.0, 9.0, 9.0, 7.0, 7.0, 5.0, 5.0, 3.0, 3.0,
                            3.0, 3.0, 17.0, 17.0, 13.0, 13.0, 11.0, 11.0, 9.0,
                            9.0, 9.0, 9.0, 7.0, 7.0, 7.0, 7.0, 5.0, 5.0, 5.0]

        exception_map = {
            RingBuffer: PySendModelFixed
        }

        num_steps = 100

        size = 1
        weights_init = np.eye(size) * 1

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
            x1_impulse=16,
            rng_seed=0
        )

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 2] = 50
        spike_raster_pre[0, 14] = 44
        spike_raster_pre[0, 28] = 34
        spike_raster_pre[0, 33] = 67
        spike_raster_pre[0, 49] = 12
        spike_raster_pre[0, 64] = 10
        spike_raster_pre[0, 82] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.DEFAULT)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        pattern_pre.s_out.connect(learning_dense.s_in)
        learning_dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(learning_dense.s_in_bap)

        monitor = Monitor()
        monitor.probe(target=learning_dense.x1, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)

    def test_stdp_graded_spike_overwrite_mode_bit_approximate(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 25.0, 20.0, 20.0, 16.0, 16.0, 13.0,
                            13.0, 10.0, 10.0, 8.0, 8.0, 6.0, 22.0, 18.0,
                            18.0, 15.0, 15.0, 13.0, 13.0, 11.0, 11.0, 9.0,
                            9.0, 7.0, 7.0, 5.0, 17.0, 13.0, 13.0, 11.0,
                            11.0, 28.0, 28.0, 23.0, 23.0, 18.0, 18.0, 14.0,
                            14.0, 12.0, 12.0, 10.0, 10.0, 8.0, 8.0, 8.0, 8.0,
                            4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                            2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 14.0, 12.0, 12.0, 10.0, 10.0, 9.0,
                            9.0, 7.0, 7.0, 7.0, 7.0, 5.0, 5.0, 5.0, 5.0, 3.0,
                            3.0, 3.0]

        exception_map = {
            RingBuffer: PySendModelFixed
        }

        num_steps = 100

        size = 1
        weights_init = np.eye(size) * 1

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
            rng_seed=0,
            x1_impulse=0
        )

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 2] = 50
        spike_raster_pre[0, 14] = 44
        spike_raster_pre[0, 28] = 34
        spike_raster_pre[0, 33] = 67
        spike_raster_pre[0, 49] = 12
        spike_raster_pre[0, 64] = 10
        spike_raster_pre[0, 82] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.OVERWRITE)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        pattern_pre.s_out.connect(learning_dense.s_in)
        learning_dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(learning_dense.s_in_bap)

        monitor = Monitor()
        monitor.probe(target=learning_dense.x1, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        import matplotlib.pyplot as plt
        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        print(list_x1_data)
        plt.figure(figsize=(12, 8))
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expecteddata")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)

    def test_stdp_graded_spike_overwrite_mode_overflow_bit_approximate(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 25.0, 20.0, 20.0, 16.0, 16.0, 13.0, 13.0,
                            10.0, 10.0, 8.0, 8.0, 6.0, 22.0, 18.0, 18.0, 15.0,
                            15.0, 13.0, 13.0, 11.0, 11.0, 9.0, 9.0, 7.0, 7.0,
                            5.0, 12.0, 10.0, 10.0, 9.0, 9.0, 28.0, 28.0, 23.0,
                            23.0, 18.0, 18.0, 14.0]

        exception_map = {
            RingBuffer: PySendModelFixed
        }

        num_steps = 40

        size = 1
        weights_init = np.eye(size) * 1

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
            x1_impulse=0,
            rng_seed=0
        )

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 2] = 50
        spike_raster_pre[0, 14] = 44
        # Overflow
        spike_raster_pre[0, 28] = 280
        spike_raster_pre[0, 33] = 67
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.OVERWRITE)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        pattern_pre.s_out.connect(learning_dense.s_in)
        learning_dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(learning_dense.s_in_bap)

        monitor = Monitor()
        monitor.probe(target=learning_dense.x1, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        import matplotlib.pyplot as plt
        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        print(list_x1_data)
        plt.figure(figsize=(12, 8))
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expecteddata")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)

    def test_stdp_graded_spike_add_saturation_mode_bit_approximate(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 25.0, 20.0, 20.0, 16.0, 16.0, 13.0,
                            13.0, 10.0, 10.0, 8.0, 8.0, 6.0, 28.0, 22.0,
                            22.0, 18.0, 18.0, 16.0, 16.0, 12.0, 12.0, 10.0,
                            10.0, 8.0, 8.0, 6.0, 23.0, 19.0, 19.0, 17.0, 17.0,
                            42.0, 42.0, 34.0, 34.0, 27.0, 27.0, 22.0, 22.0,
                            18.0, 18.0, 15.0, 15.0, 13.0, 13.0, 11.0, 11.0,
                            13.0, 13.0, 10.0, 10.0, 8.0, 8.0, 6.0, 6.0, 6.0,
                            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 11.0, 9.0, 9.0, 7.0,
                            7.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 15.0, 13.0, 13.0, 11.0, 11.0,
                            9.0, 9.0, 7.0, 7.0, 7.0, 7.0, 5.0, 5.0, 5.0, 5.0,
                            3.0, 3.0, 3.0]

        exception_map = {
            RingBuffer: PySendModelFixed
        }

        num_steps = 100

        size = 1
        weights_init = np.eye(size) * 1

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
            x1_impulse=0,
            rng_seed=0
        )

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 2] = 50
        spike_raster_pre[0, 14] = 44
        spike_raster_pre[0, 28] = 34
        spike_raster_pre[0, 33] = 67
        spike_raster_pre[0, 49] = 12
        spike_raster_pre[0, 64] = 10
        spike_raster_pre[0, 82] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.ADD_SATURATION)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        pattern_pre.s_out.connect(learning_dense.s_in)
        learning_dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(learning_dense.s_in_bap)

        monitor = Monitor()
        monitor.probe(target=learning_dense.x1, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        import matplotlib.pyplot as plt
        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        print(list_x1_data)
        plt.figure(figsize=(12, 8))
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expecteddata")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)

    def test_stdp_graded_spike_add_saturation_mode_overflow_bit_approximate(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 25.0, 20.0, 20.0, 16.0, 16.0, 13.0,
                            13.0, 10.0, 10.0, 8.0, 8.0, 6.0, 28.0, 22.0,
                            22.0, 18.0, 18.0, 16.0, 16.0, 12.0, 12.0, 10.0,
                            10.0, 8.0, 8.0, 6.0, 23.0, 19.0, 19.0, 17.0, 17.0,
                            42.0, 42.0, 34.0, 34.0, 27.0, 27.0, 22.0, 22.0,
                            18.0, 18.0, 15.0, 15.0, 13.0, 13.0, 11.0, 11.0,
                            13.0, 13.0, 10.0, 10.0, 8.0, 8.0, 6.0, 6.0, 6.0,
                            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 11.0, 9.0, 9.0, 7.0,
                            7.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 15.0, 13.0, 13.0, 11.0, 11.0,
                            9.0, 9.0, 7.0, 7.0, 7.0, 7.0, 5.0, 5.0, 5.0, 5.0,
                            3.0, 3.0, 3.0]

        exception_map = {
            RingBuffer: PySendModelFixed
        }

        num_steps = 40

        size = 1
        weights_init = np.eye(size) * 1

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
            x1_impulse=0,
            rng_seed=0
        )

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 2] = 50
        spike_raster_pre[0, 14] = 44
        spike_raster_pre[0, 28] = 254
        spike_raster_pre[0, 33] = 67
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.ADD_SATURATION)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        pattern_pre.s_out.connect(learning_dense.s_in)
        learning_dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(learning_dense.s_in_bap)

        monitor = Monitor()
        monitor.probe(target=learning_dense.x1, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        import matplotlib.pyplot as plt
        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        print(list_x1_data)
        plt.figure(figsize=(12, 8))
        plt.step(list(range(num_steps)), x1_data, label="data")
        plt.step(list(range(num_steps)), expected_x1_data, label="expecteddata")
        plt.xticks(list(range(num_steps)))
        plt.legend()
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)

    def test_stdp_graded_spike_add_no_saturation_mode_bit_approximate(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 25.0, 20.0, 20.0, 16.0, 16.0, 13.0,
                            13.0, 10.0, 10.0, 8.0, 8.0, 6.0, 28.0, 22.0,
                            22.0, 18.0, 18.0, 16.0, 16.0, 12.0, 12.0, 10.0,
                            10.0, 8.0, 8.0, 6.0, 23.0, 19.0, 19.0, 17.0, 17.0,
                            42.0, 42.0, 34.0, 34.0, 27.0, 27.0, 22.0, 22.0,
                            18.0, 18.0, 15.0, 15.0, 13.0, 13.0, 11.0, 11.0,
                            13.0, 13.0, 10.0, 10.0, 8.0, 8.0, 6.0, 6.0, 6.0,
                            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 11.0, 9.0, 9.0, 7.0,
                            7.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 15.0, 13.0, 13.0, 11.0, 11.0,
                            9.0, 9.0, 7.0, 7.0, 7.0, 7.0, 5.0, 5.0, 5.0, 5.0,
                            3.0, 3.0, 3.0]

        exception_map = {
            RingBuffer: PySendModelFixed
        }

        num_steps = 100

        size = 1
        weights_init = np.eye(size) * 1

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
            x1_impulse=0,
            rng_seed=0
        )

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 2] = 50
        spike_raster_pre[0, 14] = 44
        spike_raster_pre[0, 28] = 34
        spike_raster_pre[0, 33] = 67
        spike_raster_pre[0, 49] = 12
        spike_raster_pre[0, 64] = 10
        spike_raster_pre[0, 82] = 27
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.ADD_SATURATION)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        pattern_pre.s_out.connect(learning_dense.s_in)
        learning_dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(learning_dense.s_in_bap)

        monitor = Monitor()
        monitor.probe(target=learning_dense.x1, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        import matplotlib.pyplot as plt
        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        print(list_x1_data)
        plt.figure(figsize=(12, 8))
        plt.step(list(range(num_steps)), x1_data)
        plt.xticks(list(range(num_steps)))
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)

    def test_stdp_graded_spike_add_no_saturation_mode_overflow_bit_approximate(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 25.0, 20.0, 20.0, 16.0, 16.0, 13.0,
                            13.0, 10.0, 10.0, 8.0, 8.0, 6.0, 28.0, 22.0,
                            22.0, 18.0, 18.0, 16.0, 16.0, 12.0, 12.0, 10.0,
                            10.0, 8.0, 8.0, 6.0, 23.0, 19.0, 19.0, 17.0, 17.0,
                            42.0, 42.0, 34.0, 34.0, 27.0, 27.0, 22.0, 22.0,
                            18.0, 18.0, 15.0, 15.0, 13.0, 13.0, 11.0, 11.0,
                            13.0, 13.0, 10.0, 10.0, 8.0, 8.0, 6.0, 6.0, 6.0,
                            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 11.0, 9.0, 9.0, 7.0,
                            7.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 15.0, 13.0, 13.0, 11.0, 11.0,
                            9.0, 9.0, 7.0, 7.0, 7.0, 7.0, 5.0, 5.0, 5.0, 5.0,
                            3.0, 3.0, 3.0]

        exception_map = {
            RingBuffer: PySendModelFixed
        }

        num_steps = 40

        size = 1
        weights_init = np.eye(size) * 1

        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=4,
            A_minus=-2,
            tau_plus=10,
            tau_minus=10,
            t_epoch=2,
            x1_impulse=0,
            rng_seed=0
        )

        spike_raster_pre = np.zeros((size, num_steps))
        spike_raster_pre[0, 2] = 50
        spike_raster_pre[0, 14] = 44
        spike_raster_pre[0, 28] = 254
        spike_raster_pre[0, 33] = 67
        pattern_pre = RingBuffer(data=spike_raster_pre.astype(int))

        learning_dense = \
            LearningDense(weights=weights_init,
                          learning_rule=learning_rule,
                          name="learning_dense",
                          num_message_bits=8,
                          graded_spike_cfg=GradedSpikeCfg.ADD_SATURATION)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        pattern_pre.s_out.connect(learning_dense.s_in)
        learning_dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(learning_dense.s_in_bap)

        monitor = Monitor()
        monitor.probe(target=learning_dense.x1, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        import matplotlib.pyplot as plt
        list_x1_data = [round(data_pt, 7) for data_pt in x1_data.tolist()]
        print(list_x1_data)
        plt.figure(figsize=(12, 8))
        plt.step(list(range(num_steps)), x1_data)
        plt.xticks(list(range(num_steps)))
        plt.grid()
        plt.show()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)
