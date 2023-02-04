# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.magma.core.learning.constants import GradedSpikeCfg
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
    def test_stdp_graded_spike_default_mode_floating_point(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 0.0, 14.4773987, 14.4773987, 11.8530915,
                            11.8530915, 9.7044906, 9.7044906, 7.9453649,
                            7.9453649, 6.5051146, 6.5051146, 5.3259373,
                            5.3259373, 18.8379074, 18.8379074, 15.4231741,
                            15.4231741, 12.6274269, 12.6274269, 10.3384628,
                            10.3384628, 8.4644174, 8.4644174, 6.9300788,
                            6.9300788, 5.6738687, 5.6738687, 19.1227695,
                            19.1227695, 15.6563994, 15.6563994, 28.8183757,
                            28.8183757, 23.5944904, 23.5944904, 19.3175349,
                            19.3175349, 15.8158599, 15.8158599, 12.9489309,
                            12.9489309, 10.6016879, 10.6016879, 8.679928,
                            8.679928, 7.106524, 7.106524, 21.8183297,
                            21.8183297, 17.8633375, 17.8633375, 14.6252638,
                            14.6252638, 11.9741532, 11.9741532, 9.8036075,
                            9.8036075, 8.0265149, 8.0265149, 6.5715546,
                            6.5715546, 5.3803339, 5.3803339, 18.8824435,
                            18.8824435, 15.4596372, 15.4596372, 12.6572804,
                            12.6572804, 10.3629047, 10.3629047, 8.4844288,
                            8.4844288, 6.9464628, 6.9464628, 5.6872827,
                            5.6872827, 4.6563532, 4.6563532, 3.8122996,
                            3.8122996, 17.5986456, 17.5986456, 14.4085524,
                            14.4085524, 11.7967249, 11.7967249, 9.6583415,
                            9.6583415, 7.9075812, 7.9075812, 6.4741799,
                            6.4741799, 5.3006102, 5.3006102, 4.3397726,
                            4.3397726, 3.5531053]

        exception_map = {
            RingBuffer: PySendModelFloat
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
            x1_impulse=16
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

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=1, bias_mant=0.3)

        pattern_pre.s_out.connect(learning_dense.s_in)
        learning_dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(learning_dense.s_in_bap)

        monitor = Monitor()
        monitor.probe(target=learning_dense.x1, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)

    def test_stdp_graded_spike_overwrite_mode_floating_point(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 25.0, 20.4682688, 20.4682688, 16.7580012,
                            16.7580012, 13.7202909, 13.7202909, 11.2332241,
                            11.2332241, 9.196986, 9.196986, 7.5298553, 22.0,
                            18.0120766, 18.0120766, 14.747041, 14.747041,
                            12.073856, 12.073856, 9.8852372, 9.8852372,
                            8.0933477, 8.0933477, 6.6262727, 6.6262727,
                            5.4251332, 17.0, 13.9184228, 13.9184228,
                            11.3954408, 11.3954408, 27.4274802, 27.4274802,
                            22.4557215, 22.4557215, 18.3851898, 18.3851898,
                            15.0525203, 15.0525203, 12.3239613, 12.3239613,
                            10.0900061, 10.0900061, 8.2609983, 8.2609983,
                            6.7635334, 6.7635334, 4.9123845, 4.9123845,
                            4.0219203, 4.0219203, 3.2928698, 3.2928698,
                            2.6959738, 2.6959738, 2.2072766, 2.2072766,
                            1.8071653, 1.8071653, 1.4795818, 1.4795818,
                            1.2113791, 5.0, 4.0936538, 4.0936538, 3.3516002,
                            3.3516002, 2.7440582, 2.7440582, 2.2466448,
                            2.2466448, 1.8393972, 1.8393972, 1.5059711,
                            1.5059711, 1.2329848, 1.2329848, 1.0094826,
                            1.0094826, 0.8264944, 13.5, 11.0528652, 11.0528652,
                            9.0493206, 9.0493206, 7.4089571, 7.4089571,
                            6.065941, 6.065941, 4.9663725, 4.9663725,
                            4.0661219, 4.0661219, 3.329059, 3.329059,
                            2.725603, 2.725603, 2.231535]

        exception_map = {
            RingBuffer: PySendModelFloat
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

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)

    def test_stdp_graded_spike_add_saturation_mode_floating_point(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 25.0, 20.4682688, 20.4682688, 16.7580012,
                            16.7580012, 13.7202909, 13.7202909, 11.2332241,
                            11.2332241, 9.196986, 9.196986, 7.5298553,
                            29.5298553, 24.1770007, 24.1770007, 19.794454,
                            19.794454, 16.2063282, 16.2063282, 13.2686193,
                            13.2686193, 10.8634267, 10.8634267, 8.8942215,
                            8.8942215, 7.2819727, 24.2819727, 19.8803978,
                            19.8803978, 16.276693, 16.276693, 40.7537094,
                            40.7537094, 33.3663152, 33.3663152, 27.3180283,
                            27.3180283, 22.3661099, 22.3661099, 18.311822,
                            18.311822, 14.9924518, 14.9924518, 12.2747814,
                            12.2747814, 10.049741, 10.049741, 13.1404165,
                            13.1404165, 10.7584631, 10.7584631, 8.8082846,
                            8.8082846, 7.2116135, 7.2116135, 5.9043698,
                            5.9043698, 4.8340891, 4.8340891, 3.9578174,
                            3.9578174, 3.2403868, 8.2403868, 6.7466581,
                            6.7466581, 5.5236965, 5.5236965, 4.5224202,
                            4.5224202, 3.7026445, 3.7026445, 3.0314689,
                            3.0314689, 2.4819568, 2.4819568, 2.0320544,
                            2.0320544, 1.6637054, 1.6637054, 1.3621268,
                            14.8621268, 12.1680803, 12.1680803, 9.9623815,
                            9.9623815, 8.1565081, 8.1565081, 6.677984,
                            6.677984, 5.4674709, 5.4674709, 4.4763866,
                            4.4763866, 3.6649553, 3.6649553, 3.0006116,
                            3.0006116, 2.456693]

        exception_map = {
            RingBuffer: PySendModelFloat
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
                          graded_spike_cfg=GradedSpikeCfg.ADD_SATURATION)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        pattern_pre.s_out.connect(learning_dense.s_in)
        learning_dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(learning_dense.s_in_bap)

        monitor = Monitor()
        monitor.probe(target=learning_dense.x1, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)

    def test_stdp_graded_spike_add_no_saturation_mode_floating_point(self):
        """TODO : WRITE"""
        expected_x1_data = [0.0, 0.0, 25.0, 20.4682688, 20.4682688, 16.7580012,
                            16.7580012, 13.7202909, 13.7202909, 11.2332241,
                            11.2332241, 9.196986, 9.196986, 7.5298553,
                            29.5298553, 24.1770007, 24.1770007, 19.794454,
                            19.794454, 16.2063282, 16.2063282, 13.2686193,
                            13.2686193, 10.8634267, 10.8634267, 8.8942215,
                            8.8942215, 7.2819727, 24.2819727, 19.8803978,
                            19.8803978, 16.276693, 16.276693, 40.7537094,
                            40.7537094, 33.3663152, 33.3663152, 27.3180283,
                            27.3180283, 22.3661099, 22.3661099, 18.311822,
                            18.311822, 14.9924518, 14.9924518, 12.2747814,
                            12.2747814, 10.049741, 10.049741, 13.1404165,
                            13.1404165, 10.7584631, 10.7584631, 8.8082846,
                            8.8082846, 7.2116135, 7.2116135, 5.9043698,
                            5.9043698, 4.8340891, 4.8340891, 3.9578174,
                            3.9578174, 3.2403868, 8.2403868, 6.7466581,
                            6.7466581, 5.5236965, 5.5236965, 4.5224202,
                            4.5224202, 3.7026445, 3.7026445, 3.0314689,
                            3.0314689, 2.4819568, 2.4819568, 2.0320544,
                            2.0320544, 1.6637054, 1.6637054, 1.3621268,
                            14.8621268, 12.1680803, 12.1680803, 9.9623815,
                            9.9623815, 8.1565081, 8.1565081, 6.677984,
                            6.677984, 5.4674709, 5.4674709, 4.4763866,
                            4.4763866, 3.6649553, 3.6649553, 3.0006116,
                            3.0006116, 2.456693]

        exception_map = {
            RingBuffer: PySendModelFloat
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
                          graded_spike_cfg=GradedSpikeCfg.ADD_NO_SATURATION)

        lif_1 = LIF(shape=(size,), du=0, dv=0, vth=10000, bias_mant=20000)

        pattern_pre.s_out.connect(learning_dense.s_in)
        learning_dense.a_out.connect(lif_1.a_in)
        lif_1.s_out.connect(learning_dense.s_in_bap)

        monitor = Monitor()
        monitor.probe(target=learning_dense.x1, num_steps=num_steps)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt",
                               exception_proc_model_map=exception_map)
        run_cnd = RunSteps(num_steps=num_steps)

        pattern_pre.run(condition=run_cnd, run_cfg=run_cfg)

        x1_data = monitor.get_data()["learning_dense"]["x1"][:, 0]

        pattern_pre.stop()

        np.testing.assert_almost_equal(x1_data, expected_x1_data)


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
