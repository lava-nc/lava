# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class ConvInTime(AbstractProcess):
    """Connection Process that mimics a convolution of the incoming
    events/spikes with a kernel in the time dimension. Realizes the following abstract
    behavior: a_out[t] = weights[t-1] * s_in[t-1] + weights[t] * s_in[t] + weights[t+1] * s_in[t+1]

    Parameters
    ----------
    weights : numpy.ndarray
        3D connection weight matrix of form (kernel_size, num_flat_output_neurons,
        num_flat_input_neurons) in C-order (row major).

    weight_exp : int, optional
        Shared weight exponent of base 2 used to scale magnitude of
        weights, if needed. Mostly for fixed point implementations.
        Unnecessary for floating point implementations.
        Default value is 0.

    num_weight_bits : int, optional
        Shared weight width/precision used by weight. Mostly for fixed
        point implementations. Unnecessary for floating point
        implementations.
        Default is for weights to use full 8 bit precision.

    sign_mode : SignMode, optional
        Shared indicator whether synapse is of type SignMode.NULL,
        SignMode.MIXED, SignMode.EXCITATORY, or SignMode.INHIBITORY. If
        SignMode.MIXED, the sign of the weight is
        included in the weight bits and the fixed point weight used for
        inference is scaled by 2.
        Unnecessary for floating point implementations.

        In the fixed point implementation, weights are scaled according to
        the following equations:
        w_scale = 8 - num_weight_bits + weight_exp + isMixed()
        weights = weights * (2 ** w_scale)

    num_message_bits : int, optional
        Determines whether the Dense Process deals with the incoming
        spikes as binary spikes (num_message_bits = 0) or as graded
        spikes (num_message_bits > 0). Default is 0.
        """
    def __init__(self,
                 *,
                 weights: np.ndarray,
                 name: ty.Optional[str] = None,
                 num_message_bits: ty.Optional[int] = 0,
                 log_config: ty.Optional[LogConfig] = None,
                 **kwargs) -> None:

        super().__init__(weights=weights,
                         num_message_bits=num_message_bits,
                         name=name,
                         log_config=log_config,
                         **kwargs)

        self._validate_weights(weights)
        shape = weights.shape # [kernel_size, n_flat_output_neurons, n_flat_input_neurons]
        # Ports
        self.s_in = InPort(shape=(shape[2],))
        self.a_out = OutPort(shape=(shape[1],))

        # Variables
        self.weights = Var(shape=shape, init=weights)
        self.a_buff = Var(shape=(shape[1], shape[0]), init=0)
        self.num_message_bits = Var(shape=(1,), init=num_message_bits)

    @staticmethod
    def _validate_weights(weights: np.ndarray) -> None:
        if len(np.shape(weights)) != 3:
            raise ValueError("Dense Process 'weights' expects a 3D matrix, "
                             f"got {weights}.")


########## validation code ##########
# from lava.proc.conv_in_time.process import ConvInTime
# from lava.proc import io
# import numpy as np
# import torch
# import torch.nn as nn
# np.set_printoptions(linewidth=np.inf)

# num_steps = 10
# n_flat_input_neurons = 2
# n_flat_output_neurons = 3
# kernel_size = 3
# input = np.random.choice([0, 1], size=(n_flat_input_neurons, num_steps))
# sender = io.source.RingBuffer(data=input)
# weights = np.random.rand(kernel_size, n_flat_output_neurons, n_flat_input_neurons)
# conv_in_time = ConvInTime(weights=weights, name='conv_in_time')
# receiver = io.sink.RingBuffer(shape=(n_flat_output_neurons,), buffer=num_steps)

# sender.s_out.connect(conv_in_time.s_in)
# conv_in_time.a_out.connect(receiver.a_in)

# from lava.magma.core.run_conditions import RunSteps
# from lava.magma.core.run_configs import Loihi1SimCfg

# run_condition = RunSteps(num_steps=num_steps)
# run_cfg = Loihi1SimCfg(select_tag="floating_pt")

# conv_in_time.run(condition=run_condition, run_cfg=run_cfg)
# output = receiver.data.get()
# conv_in_time.stop()

# tensor_input = torch.tensor(input, dtype=torch.float32)
# tensor_weights = torch.tensor(weights, dtype=torch.float32)
# conv_layer = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=3, bias=False)
# conv_layer.weight = nn.Parameter(tensor_weights.permute(1, 2, 0))
# torch_output = conv_layer(tensor_input.unsqueeze(0)).squeeze(0).detach().numpy()

# print(' lava output: ', output)
# print('torch output: ', torch_output)