# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.learning.learning_rule import Loihi2FLearningRule
from lava.magma.core.learning.utils import float_to_literal


class STDPLoihi(Loihi2FLearningRule):

    def __init__(
            self,
            learning_rate: float,
            A_plus: float,
            A_minus: float,
            tau_plus: float,
            tau_minus: float,
            *args,
            **kwargs
    ):
        """
        Spike-timing dependent plasticity (STDP) as defined in
        Gerstner and al. 1996.

        Parameters
        ==========

        learning_rate: float
            Overall learning rate scaling the intensity of weight changes.
        A_plus:
            Scaling the weight change on pre-synaptic spike times.
        A_minus:
            Scaling the weight change on post-synaptic spike times.
        tau_plus:
            Time constant of the pre-synaptic activity trace.
        tau_minus:
            Time constant of the post-synaptic activity trace.

        """

        self.learning_rate = float_to_literal(learning_rate)
        self.A_plus = float_to_literal(A_plus)
        self.A_minus = float_to_literal(A_minus)
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        # String learning rule for dw
        dw = f"{self.learning_rate} * {self.A_minus} * x0 * y1 +" \
             f"{self.learning_rate} * {self.A_plus} * y0 * x1"

        # Other learning-related parameters
        # Trace impulse values
        x1_impulse = kwargs.pop("x1_impulse", 16)
        y1_impulse = kwargs.pop("y1_impulse", 16)

        # Trace decay constants
        x1_tau = tau_plus
        y1_tau = tau_minus

        super().__init__(
            dw=dw,
            x1_impulse=x1_impulse,
            x1_tau=x1_tau,
            y1_impulse=y1_impulse,
            y1_tau=y1_tau,
            *args,
            **kwargs
        )
