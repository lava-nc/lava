# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

# Modified from src/lava/proc/dense/process.py 
# Modified by Kevin Sargent, Pennsylvania State University 

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

class Delay(AbstractProcess):
    """Modified version of Dense connections to include synaptic delays.
    Stores the input spikes in a buffer Sb of size (num_input_neurons, maximum_delay_length)
    The output for each timestep is calculated as:
        a_out[j] = sum( W[j, i] * Sb[i, D[j, i]] ), 
    where the sum is over the inputs i and j are the output indices.

    Parameters
    ----------

    weights (W):
    2D Connection weight matrix of the form (num_flat_output_neurons,
    num_flat_input_neurons) in C-order (row major). 

    delays (D):
    2D matrix of the form (num_flat_output_neurons, 
    num_flat_input_neurons) in C-order (row major),
    where the element D[j, i] is the delay between input
    i and output j. 

    use_graded_spike: bool
    flag to indicated graded spike. Default is False

    Other Variables
    ---------------

    s_buff (Sb):
    2D matrix of form (num_flat_input_neurons, max_delay_length) 
    which stores the input spikes for the max_delay_length previous
    time steps.
    
    """

    # ToDo: (KS) Implement the parameters weight_exp, num_weight_bits, and sign_mode
    # from the Dense process
    
    # ToDo: (KS) Implement a process model that is bit-accurate to Loihi

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        if len(shape) != 2:
            raise AssertionError("Dense Process 'shape' expected a 2D tensor.")
        weights = kwargs.pop("weights", np.zeros(shape=shape))
        if len(np.shape(weights)) != 2:
            raise AssertionError("Dense Process 'weights' expected a 2D "
                                 "matrix.")
        # weight_exp = kwargs.pop("weight_exp", 0)
        # num_weight_bits = kwargs.pop("num_weight_bits", 8)
        # sign_mode = kwargs.pop("sign_mode", 1)
        use_graded_spike = kwargs.get('use_graded_spike', False)
        delays = kwargs.pop("delays", np.zeros(shape=shape, dtype=int))
        if len(np.shape(delays)) != 2: 
            raise AssertionError("Dense Process 'delays' expected a 2D matrix.")
        max_delay = int(np.max(delays))
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, init=weights)
        # self.weight_exp = Var(shape=(1,), init=weight_exp)
        # self.num_weight_bits = Var(shape=(1,), init=num_weight_bits)
        # self.sign_mode = Var(shape=(1,), init=sign_mode)
        self.s_buff = Var(shape=(shape[1],max_delay+1), init=False)
        self.use_graded_spike = Var(shape=(1,), init=use_graded_spike)
        self.delays = Var(shape=shape, init=delays)
        self.max_delay = Var(shape=(1,), init=max_delay)
