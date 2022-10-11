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


class STDPLoihi(LoihiLearningRule):

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

        self.learning_rate = learning_rate
        self.A_plus = str(A_plus) if A_plus > 0 else f"({str(A_plus)})"
        self.A_minus = str(A_minus) if A_minus > 0 else f"({str(A_minus)})"
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        # String learning rule for dw
        dw = f"{self.learning_rate} * {self.A_plus} * x0 * y1 +" \
             f"{self.learning_rate} * {self.A_minus} * y0 * x1"

        # Other learning-related parameters
        # Trace impulse values
        x1_impulse = kwargs.get("x1_impulse", 16)
        y1_impulse = kwargs.get("y1_impulse", 16)

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
