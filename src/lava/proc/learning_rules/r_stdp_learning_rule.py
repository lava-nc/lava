# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2022 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
# See: https://spdx.org/licenses/

from lava.magma.core.learning.learning_rule import LoihiLearningRule
from lava.magma.core.learning.utils import float_to_literal


class RewardModulatedSTDP(LoihiLearningRule):
    def __init__(
            self,
            learning_rate: float,
            A_plus: float,
            A_minus: float,
            pre_trace_decay_tau: float,
            post_trace_decay_tau: float,
            pre_trace_kernel_magnitude: float,
            post_trace_kernel_magnitude: float,
            eligibility_trace_decay_tau: float,
            *args,
            **kwargs
    ):
        """
        Reward-Modulated Spike-timing dependent plasticity (STDP)
        as defined in Frémaux, Nicolas, and Wulfram Gerstner.
        "Neuromodulated spike-timing-dependent plasticity, and
        theory of three-factor learning rules." Frontiers in
        neural circuits 9 (2016).

        Parameters
        ==========

        learning_rate: float
            Overall learning rate scaling the intensity of weight changes.
        A_plus:
            Scaling the weight change on pre-synaptic spike times.
        A_minus:
            Scaling the weight change on post-synaptic spike times.
        pre_trace_decay_tau:
            Decay time constant of the pre-synaptic activity trace.
        post_trace_decay_tau:
            Decay time constant of the post-synaptic activity trace.
        pre_trace_kernel_magnitude:
            The magnitude of increase to the pre-synaptic trace value
            at the instant of pre-synaptic spike.
        post_trace_kernel_magnitude:
            The magnitude of increase to the post-synaptic trace value
            at the instant of post-synaptic spike.
        eligibility_trace_decay_tau:
            Decay time constant of the eligibility trace.

        """
        self.learning_rate = float_to_literal(learning_rate) 
        self.A_plus = str(A_plus) if A_plus > 0 else f"({str(A_plus)})"
        self.A_minus = str(A_minus) if A_minus > 0 else f"({str(A_minus)})"
        self.pre_trace_decay_tau = pre_trace_decay_tau
        self.post_trace_decay_tau = post_trace_decay_tau
        self.pre_trace_kernel_magnitude = pre_trace_kernel_magnitude
        self.post_trace_kernel_magnitude = post_trace_kernel_magnitude
        self.eligibility_trace_decay_tau = float_to_literal(eligibility_trace_decay_tau)

        # Trace impulse values
        x1_impulse = pre_trace_kernel_magnitude
        y1_impulse = post_trace_kernel_magnitude
        y2_impulse = 0

        # Trace decay constants
        x1_tau = self.pre_trace_decay_tau
        y1_tau = self.post_trace_decay_tau
        y2_tau = 2 ** 32 - 1

        # Eligibility trace represented as dt
        dt = f"{self.learning_rate} * {self.A_plus} * x0 * y1 +" \
             f"{self.learning_rate} * {self.A_minus} * y0 * x1 -" \
             f"u0 * t * {self.eligibility_trace_decay_tau}"

        # Reward-modulated weight update
        dw = " u0 * t * y2 "

        super().__init__(
            dw=dw,
            dt=dt,
            x1_impulse=x1_impulse,
            x1_tau=x1_tau,
            y1_impulse=y1_impulse,
            y1_tau=y1_tau,
            y2_impulse=y2_impulse,
            y2_tau=y2_tau,
            *args,
            **kwargs
        )
