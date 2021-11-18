# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.proc.conv import utils


class Conv(AbstractProcess):
    """Convolution connections between neurons.

    Parameters
    ----------
    input_shape : tuple of three ints
        shape of input to the process in (X, Y, Z) or (W, H, C) format.
    weight : tensor/array
        convolution kernel weight. The dimension should be in
        (C_out, W, H, C_in) format.
    padding : int or tuple of two ints
        convolution padding size. Default is 0.
    stride : int or tuple of two ints
        convolution stride. Default is 1.
    dilation : int or tuple of two ints
        convolution dilation. Default is 1.
    groups : int
        number of groups in convolution. Default is 1.

    Note
    ----
    padding, stride and dilation are expected in (X, Y) or (W, H) if tuple.
    """
    def __init__(self, **kwargs):
        # the process In/OutPort shapes are considered to be XYZ(WHC) format.
        # the kernel weight shape is expected to be in (C_out, W, H, C_in)
        # format.
        # Why? This is the format that Loihi conv feature uses.
        super().__init__(**kwargs)

        def broadcast_arg(name, default):
            shape = kwargs.get(name, default)
            if np.isscalar(shape):
                return (shape, shape)
            elif len(shape) == 1:
                return (shape[0], shape[0])
            elif len(shape) == 2:
                return (shape[0], shape[1])
            else:
                raise Exception(
                    f'Expected {name} to be two dimensional.'
                    f'Found {name} = {shape}.'
                )

        input_shape = kwargs.get('input_shape', (1, 1, 1))
        kernel_size = kwargs['weight'].shape[1:3]
        in_channels = input_shape[-1]
        out_channels = kwargs['weight'].shape[0]
        padding = broadcast_arg('padding', 0)
        stride = broadcast_arg('stride', 1)
        dilation = broadcast_arg('dilation', 1)
        groups = kwargs.get('groups', 1)

        if len(input_shape) != 3:
            raise Exception(
                f'Expected input shape to be 3 dimensional.'
                f'Found {input_shape}.'
            )
        if not np.isscalar(groups):
            raise Exception(
                f'Expected groups to be a scalar.'
                f'found {groups = }.'
            )
        if in_channels % groups != 0:
            raise Exception(
                f'Expected number of in_channels to be divisible by groups.'
                f'Found {in_channels = } and {groups = }.'
            )
        if out_channels % groups != 0:
            raise Exception(
                f'Expected number of out_channels to be divisible by groups.'
                f'Found {out_channels = } and {groups = }.'
            )

        output_shape = utils.output_shape(
            input_shape, out_channels, kernel_size, stride, padding, dilation
        )

        self.output_shape = output_shape
        self.input_shape = input_shape
        self.s_in = InPort(shape=input_shape)
        self.a_out = OutPort(shape=output_shape)
        self.weight = Var(
            shape=kwargs['weight'].shape,
            init=kwargs.pop('weight')
        )
        self.kernel_size = Var(shape=(2,), init=kernel_size)
        self.padding = Var(shape=(2,), init=padding)
        self.stride = Var(shape=(2,), init=stride)
        self.dilation = Var(shape=(2,), init=dilation)
        self.groups = Var(shape=(1,), init=groups)
