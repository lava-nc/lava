# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
import typing as ty
from lava.proc.lif.process import LIF, AbstractLIF, LogConfig, LearningLIF
from lava.proc.dense.process import LearningDense, Dense
from lava.magma.core.process.neuron import LearningNeuronProcess
from lava.proc.learning_rules.r_stdp_learning_rule import RewardModulatedSTDP
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.neuron import LearningNeuronModelFloat
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.lif.models import AbstractPyLifModelFloat
from lava.proc.io.source import RingBuffer as SpikeIn


# class RSTDPLIF(LearningNeuronProcess, AbstractLIF):
#     """Leaky-Integrate-and-Fire (LIF) neural Process with RSTDP learning rule.
#
#     Parameters
#     ----------
#     shape : tuple(int)
#         Number and topology of LIF neurons.
#     u : float, list, numpy.ndarray, optional
#         Initial value of the neurons' current.
#     v : float, list, numpy.ndarray, optional
#         Initial value of the neurons' voltage (membrane potential).
#     du : float, optional
#         Inverse of decay time-constant for current decay. Currently, only a
#         single decay can be set for the entire population of neurons.
#     dv : float, optional
#         Inverse of decay time-constant for voltage decay. Currently, only a
#         single decay can be set for the entire population of neurons.
#     bias_mant : float, list, numpy.ndarray, optional
#         Mantissa part of neuron bias.
#     bias_exp : float, list, numpy.ndarray, optional
#         Exponent part of neuron bias, if needed. Mostly for fixed point
#         implementations. Ignored for floating point implementations.
#     vth : float, optional
#         Neuron threshold voltage, exceeding which, the neuron will spike.
#         Currently, only a single threshold can be set for the entire
#         population of neurons.
#
#     """
#
#     def __init__(
#         self,
#         *,
#         shape: ty.Tuple[int, ...],
#         u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
#         v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
#         du: ty.Optional[float] = 0,
#         dv: ty.Optional[float] = 0,
#         bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
#         bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
#         vth: ty.Optional[float] = 10,
#         name: ty.Optional[str] = None,
#         log_config: ty.Optional[LogConfig] = None,
#         learning_rule: RewardModulatedSTDP = None,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             shape=shape,
#             u=u,
#             v=v,
#             du=du,
#             dv=dv,
#             bias_mant=bias_mant,
#             bias_exp=bias_exp,
#             name=name,
#             log_config=log_config,
#             learning_rule=learning_rule,
#             **kwargs,
#         )
#         self.vth = Var(shape=(1,), init=vth)
#
#         #self.a_graded_reward_in = InPort(shape=shape)


@implements(proc=LearningLIF, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyRSTDPLIFModel(LearningNeuronModelFloat, AbstractPyLifModelFloat):
    """Implementation of Leaky-Integrate-and-Fire neural
    process in floating point precision with learning enabled.
    """

    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.s_out_buff = np.zeros(proc_params["shape"])

    def spiking_activation(self):
        """Spiking activation function for Learning LIF."""
        return self.v > self.vth

    def filter_spike_trains(self):
        """Filters the spike trains with an exponential filter."""

        y1_impulse = self.proc_params[
            "learning_rule"
        ].post_trace_kernel_magnitude
        y1_tau = self.proc_params["learning_rule"].post_trace_decay_tau

        return self.y1 * np.exp(-1 / y1_tau) + self.s_out_buff * y1_impulse

    def calculate_third_factor_trace(self, s_graded_in: float) -> float:
        """Generate's a third factor Reward traces based on
        Graded input spikes to the Learning LIF process.

        Currently, the third factor resembles the input graded spike.
        """
        return s_graded_in

    def run_spk(self) -> None:
        """Calculates the third factor trace and sends it to the
        Dense process for learning.
        """
        self.y1 = self.filter_spike_trains()

        super().run_spk()

        a_graded_in = self.a_in_reward.recv()

        self.y2 = self.calculate_third_factor_trace(a_graded_in)
        self.y3 = self.y3.astype(bool) | self.s_out_buff.astype(bool)

        self.s_out_y1.send(self.y1)
        self.s_out_y2.send(self.y2)
        self.s_out_y3.send(self.y3)

        if self.time_step % self.proc_params["learning_rule"].t_epoch == 0:
            self.y3 = self.y3 & False


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
        np.testing.assert_almost_equal(weight_after_run, np.array([[60]]))

    def test_stdp_fixed_point_multi_synapse(self):
        """Known value test. Run a simple learning dense layer between two LIF
        population with multiple neurons and compare to the resulting weight
        from previous runs."""
        learning_rule = STDPLoihi(
            learning_rate=1,
            A_plus=-2,
            A_minus=1,
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
            A_plus=-2,
            A_minus=1,
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

    def test_rstdp_floating_point(self):
        """Known value test. Run a simple learning dense layer between two LIF
        and compare to the resulting weight from previous runs."""

        learning_rule = RewardModulatedSTDP(
            learning_rate=1,
            A_plus=-2,
            A_minus=2,
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

        lif_1 = LearningLIF(
            shape=(size,),
            du=0,
            dv=0,
            vth=1,
            bias_mant=0.15,
            learning_rule=learning_rule,
        )

        # reward
        reward_signal = np.zeros((size, num_steps))
        reward_signal[:, num_steps // 3 : num_steps // 2] = 1

        reward = SpikeIn(data=reward_signal.astype(float))
        reward_conn = Dense(weights=np.eye(size))
        reward.s_out.connect(reward_conn.s_in)
        reward_conn.a_out.connect(lif_1.a_in_reward)

        lif_0.s_out.connect(dense.s_in)
        dense.a_out.connect(lif_1.a_in)

        # Connect traces from LIF to Dense
        # y1: exp filtered spike trains
        # y2: reward
        # y3: spikes (BAP)
        lif_1.s_out_y1.connect(dense.s_in_y1)
        lif_1.s_out_y2.connect(dense.s_in_y2)
        lif_1.s_out_y3.connect(dense.s_in_y3)

        run_cfg = Loihi2SimCfg(select_tag="floating_pt", exception_proc_model_map={LearningLIF: PyRSTDPLIFModel})
        run_cnd = RunSteps(num_steps=num_steps)
        weight_before_run = dense.weights.get()

        lif_0.run(condition=run_cnd, run_cfg=run_cfg)

        weight_after_run = dense.weights.get()
        lif_0.stop()

        np.testing.assert_almost_equal(weight_before_run, weights_init)
        np.testing.assert_almost_equal(
            weight_after_run, np.array([[-62.706254]])
        )
