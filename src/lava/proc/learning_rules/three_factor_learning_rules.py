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


class GatedHebbianLoihi(LoihiLearningRule):
    def __init__(
            self,
            learning_rate: float,
            tau_plus: float,
            tau_minus: float,
            *args,
            **kwargs
    ):
        """
        Gated hebbian learning rule.

        dw = learning_rate * e * x * y

        x and y are the pre- resp. post-synaptic low-pass filtered
        spike trains. The error signal e is provided by the
        post-synaptic neuron as second trace (y2).
        """

        self.learning_rate = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        # String learning rule for dw
        dw = f"{self.learning_rate} * u0 * x1 * y1 * y2 "

        # Other learning-related parameters
        # Trace impulse values
        x1_impulse = 16
        y1_impulse = 16

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
