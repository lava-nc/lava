# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class Dense(AbstractProcess):
    """Dense connections between neurons. Realizes the      following abstract
    behavior: a_out = weights * s_in '

    Parameters
    ----------

    weights:
    2D Connection weight matrix of form (num_flat_output_neurons,
    num_flat_input_neurons) in C-order (row major).

    weight_exp:
    Shared weight exponent used to
    scale magnitude of weights, if needed. Mostly for fixed point
    implementations. Unnecessary for floating point implementations.
    Default value is 0.

    num_weight_bits:
    Shared weight width/precision used by weight. Mostly for
    fixed point  implementations. Unnecessary for floating point
    implementations.
    Default is for weights to use full 8 bit precision.

    sign_mode:
    Shared indicator whether synapse is of 'null' (0),
    ’mixed’ (1, default), ’excitatory’ (2) or  ’inhibitory’ (3) type. If
    ’mixed’, the sign of the weight is included in the weight bits and
    the fixed point weight used for inference is scaled by 2.
    Unnecessary for floating point implementations.

    In the fixed point implementation, weights are scaled according to the
    following equations:
    weights = weights * (2 ** w_scale)
    w_scale =  8 - num_weight_bits + weight_exp + isMixed()

    a_buff:
    Circular buffer that stores output activations accumulated in current
    timestep for future timesteps.

    """

    # ToDo: (DR) Implement a ProcModel that supports synaptic delays. a_buff
    # must then be adjusted to the length of the delay.

    # ToDo: (DR) Revisit the implementation of w_scale so that less of this
    # computation is exposed at the level of the Process.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        if len(shape) != 2:
            raise AssertionError("Dense Process 'shape' expected a 2D tensor.")
        weights = kwargs.pop("weights", np.zeros(shape=shape))
        if len(np.shape(weights)) != 2:
            raise AssertionError("Dense Process 'weights' expected a 2D "
                                 "matrix.")
        weight_exp = kwargs.pop("weight_exp", 0)
        num_weight_bits = kwargs.pop("num_weight_bits", 8)
        sign_mode = kwargs.pop("sign_mode", 1)

        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, init=weights)
        self.weight_exp = Var(shape=(1,), init=weight_exp)
        self.num_weight_bits = Var(shape=(1,), init=num_weight_bits)
        self.sign_mode = Var(shape=(1,), init=sign_mode)
        self.a_buff = Var(shape=(shape[0],), init=0)
