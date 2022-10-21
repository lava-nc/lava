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


class DoubleExponentialSuperSpikeLoihi(LoihiLearningRule):
    def __init__(
            self,
            learning_rate: str,
            pre_synaptic_decay_tau: float,
            pre_synaptic_rise_tau: float,
            eligibility_trace_decay_tau: str,
            eligibility_trace_rise_tau: str,
            *args,
            **kwargs
    ):
        self.learning_rate = learning_rate
        self.pre_synaptic_decay_tau = pre_synaptic_decay_tau
        self.pre_synaptic_rise_tau = pre_synaptic_rise_tau
        self.eligibility_trace_decay_tau = eligibility_trace_decay_tau
        self.eligibility_trace_rise_tau = eligibility_trace_rise_tau

        sign_delay = -1

        x1_tau = pre_synaptic_decay_tau 
        x2_tau = pre_synaptic_rise_tau

        # error signal decay constant 
        y2_tau = 2 ** 32-1

        # surrogate gradient of membrane potential u, decay constant
        # (y3 == surrogate gradient)
        y3_tau = 2 ** 32-1 
        
        # Impulses
        x1_impulse = 1/((1/x2_tau) - (1/x1_tau))
        x2_impulse = 1/((1/x2_tau) - (1/x1_tau))

        # Zero impulse value for error and surrogate gradients. 
        y2_impulse = 0
        y3_impulse = 0

        dt = f"u0 * y3 * x1 - u0 * y3 * x2 - u0 * {eligibility_trace_rise_tau} * t"

        dd = f"t * u0 - u0 * {eligibility_trace_decay_tau} * t"

        dw = f"{learning_rate} * u0 * y2 * d"

        super().__init__(
            dw=dw,
            dd=dd,
            dt=dt,
            x1_impulse=x1_impulse,
            x1_tau=x1_tau,
            x2_impulse=x2_impulse,
            x2_tau=x2_tau,
            y2_impulse=y2_impulse,
            y2_tau=y2_tau,
            y3_impulse=y3_impulse,
            y3_tau=y3_tau,
            *args,
            **kwargs
        )


class SingleExponentialSuperSpikeLoihi(LoihiLearningRule):
    def __init__(
            self,
            learning_rate: str,
            pre_synaptic_decay_tau: float,
            eligibility_trace_decay_tau: str,
            *args,
            **kwargs
    ):
        self.learning_rate = learning_rate
        self.pre_synaptic_decay_tau = pre_synaptic_decay_tau
        self.eligibility_trace_decay_tau = eligibility_trace_decay_tau

        x1_tau = pre_synaptic_decay_tau 

        # error signal decay constant
        # (y2 == surrogate gradient) 
        y2_tau = 2 ** 32-1

        # surrogate gradient of membrane potential u, decay constant
        # (y3 == surrogate gradient)
        y3_tau = 2 ** 32-1 
        
        # Impulses
        x1_impulse = 1

        # Zero impulse value for error and surrogate gradients. 
        y2_impulse = 0
        y3_impulse = 0

        dt = f"u0 * y3 * x1 - u0 * {eligibility_trace_decay_tau} * t"

        dw = f"{learning_rate} * u0 * y2 * t"

        super().__init__(
            dw=dw,
            dt=dt,
            x1_impulse=x1_impulse,
            x1_tau=x1_tau,
            y2_impulse=y2_impulse,
            y2_tau=y2_tau,
            y3_impulse=y3_impulse,
            y3_tau=y3_tau,
            *args,
            **kwargs
        )