# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
from lava.proc.lif.process import LIF, LearningLIF
from lava.proc.dense.process import LearningDense, Dense
from lava.proc.learning_rules.r_stdp_learning_rule import RewardModulatedSTDP
from lava.magma.core.model.py.neuron import (
    LearningNeuronModelFloat,
    LearningNeuronModelFixed,
)
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
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
        np.testing.assert_almost_equal(weight_after_run, np.array([[56]]))

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
            weight_after_run, np.array([[31.0, 2.0, -24.0],
                                        [-64.0, 30.0, 14.0]])
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
        np.testing.assert_almost_equal(weight_after_run, np.array([[56]]))


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
            weight_after_run, np.array([[33.4210359]])
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
        lif_1.s_out_y3.connect(dense.s_in_y3)

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
                    [191.7300724, 31.3616088, 255.5749675],
                    [187.6922553, 17.4506295, 250.7446092]
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
        lif_1.s_out_y3.connect(dense.s_in_y3)

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
        lif_1.s_out_y3.connect(dense.s_in_y3)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(
            weight_after_run, np.array([[2.0, 1.0, -1.0], [15.0, 20.0, 9.0]])
        )
