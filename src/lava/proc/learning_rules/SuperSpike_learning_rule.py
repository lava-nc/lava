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

from lava.magma.core.learning.learning_rule import Loihi3FLearningRule
from lava.magma.core.learning.utils import float_to_literal


class DoubleExponentialSuperSpikeLoihi(Loihi3FLearningRule):
    def __init__(
            self,
            learning_rate: float,
            pre_synaptic_decay_tau: float,
            pre_synaptic_rise_tau: float,
            eligibility_trace_decay_tau: float,
            eligibility_trace_rise_tau: float,
            *args,
            **kwargs
    ):
        self.learning_rate = float_to_literal(learning_rate)
        self.pre_synaptic_decay_tau = pre_synaptic_decay_tau
        self.pre_synaptic_rise_tau = pre_synaptic_rise_tau
        self.eligibility_trace_decay_tau = float_to_literal(eligibility_trace_decay_tau)
        self.eligibility_trace_rise_tau = float_to_literal(eligibility_trace_rise_tau)

        x1_tau = pre_synaptic_decay_tau 
        x2_tau = pre_synaptic_rise_tau
        
        # Impulses
        x1_impulse = 1/((1/x2_tau) - (1/x1_tau))
        x2_impulse = 1/((1/x2_tau) - (1/x1_tau))

        dt = f"u0 * y3 * x1 - u0 * y3 * x2 - u0 * {self.eligibility_trace_rise_tau} * t"

        dd = f"t * u0 - u0 * {self.eligibility_trace_decay_tau} * d"

        dw = f"{self.learning_rate} * u0 * y2 * d"

        super().__init__(
            dw=dw,
            dd=dd,
            dt=dt,
            x1_impulse=x1_impulse,
            x1_tau=x1_tau,
            x2_impulse=x2_impulse,
            x2_tau=x2_tau,
            *args,
            **kwargs
        )


class SingleExponentialSuperSpikeLoihi(Loihi3FLearningRule):
    def __init__(
            self,
            learning_rate: float,
            pre_synaptic_decay_tau: float,
            eligibility_trace_decay_tau: float,
            *args,
            **kwargs
    ):
        self.learning_rate = float_to_literal(learning_rate)
        self.pre_synaptic_decay_tau = pre_synaptic_decay_tau
        self.eligibility_trace_decay_tau = float_to_literal(eligibility_trace_decay_tau)

        x1_tau = pre_synaptic_decay_tau 
        
        # Impulses
        x1_impulse = 1

        dt = f"u0 * y3 * x1 - u0 * {self.eligibility_trace_decay_tau} * t"

        dw = f"{self.learning_rate} * u0 * y2 * t"

        super().__init__(
            dw=dw,
            dt=dt,
            x1_impulse=x1_impulse,
            x1_tau=x1_tau,
            *args,
            **kwargs
        )