# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np

from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.proc.conv import utils


class Conv(AbstractProcess):
    def __init__(
            self,
            *,
            weight: np.ndarray,
            weight_exp: ty.Optional[int] = 0,
            input_shape: ty.Optional[ty.Tuple[int, int, int]] = (1, 1, 1),
            padding: ty.Optional[ty.Union[int, ty.Tuple[int, int]]] = 0,
            stride: ty.Optional[ty.Union[int, ty.Tuple[int, int]]] = 1,
            dilation: ty.Optional[ty.Union[int, ty.Tuple[int, int]]] = 1,
            groups: ty.Optional[int] = 1,
            num_weight_bits: ty.Optional[int] = 8,
            num_message_bits: ty.Optional[int] = 0,
            name: ty.Optional[str] = None,
            log_config: ty.Optional[LogConfig] = None) -> None:
        """Connection Process that mimics a convolution of the incoming
        events/spikes with a kernel of synaptic weights.

        Parameters
        ----------
        weight : numpy.ndarray
            Weights of the convolutional kernel; the format of the array
            should be four-dimensional, with the shape (C_out, W, H, C_in),
            where W and H are the width and height of the kernel in spatial
            coordinates (e.g., image pixels), and C_out and C_in are the number
            of outgoing and incoming channels/filters.
        weight_exp : int, optional
            Shared weight exponent of base-2 used to scale the magnitude of
            weights, if needed. The effective weight is set to
            >>> weight * pow(2, weight_exp)
            This parameter is mostly needed for fixed point
            implementations and unnecessary for floating point implementations.
            Default value is 0.
        input_shape : tuple(int, int, int), optional
            Shape of input to the process in (X, Y, Z) or (W, H, C) format.
            See 'weight'.
        padding : int or tuple(int, int), optional
            Convolution padding size. Default is 0.
        stride : int or tuple(int, int)
            Convolution stride. Default is 1.
        dilation : int or tuple(int, int)
            Convolution dilation. Default is 1.
        groups : int
            Number of groups in convolution. Default is 1.
        num_weight_bits: int
            Shared weight width/precision used by weight. Mostly for
            fixed point  implementations. Unnecessary for floating point
            implementations.
            Default is for weights to use full 8 bit precision.
        num_message_bits: int
            Number of spike message bits. Default is 0.

        Note
        ----
        Padding, stride and dilation are expected in (X, Y) or (W,
        H) if they are supplied as a tuple.
        """

        super().__init__(weight=weight,
                         weight_exp=weight_exp,
                         input_shape=input_shape,
                         padding=padding,
                         stride=stride,
                         dilation=dilation,
                         groups=groups,
                         num_weight_bits=num_weight_bits,
                         num_message_bits=num_message_bits,
                         name=name,
                         log_config=log_config)

        self._validate_input_shape(input_shape)
        self._validate_groups(groups)

        in_channels = input_shape[-1]
        out_channels = weight.shape[0]
        self._validate_channels(in_channels, out_channels, groups)

        kernel_size = weight.shape[1:3]
        padding = utils.make_tuple(padding)
        stride = utils.make_tuple(stride)
        dilation = utils.make_tuple(dilation)

        output_shape = utils.output_shape(
            input_shape, out_channels, kernel_size, stride, padding, dilation
        )

        self.output_shape = output_shape
        self.input_shape = input_shape
        self.s_in = InPort(shape=input_shape)
        self.a_out = OutPort(shape=output_shape)
        self.weight = Var(shape=weight.shape, init=weight)
        self.weight_exp = Var(shape=(1,), init=weight_exp)
        self.num_weight_bits = Var(shape=(1,), init=num_weight_bits)
        self.kernel_size = Var(shape=(2,), init=kernel_size)
        self.padding = Var(shape=(2,), init=padding)
        self.stride = Var(shape=(2,), init=stride)
        self.dilation = Var(shape=(2,), init=dilation)
        self.groups = Var(shape=(1,), init=groups)
        self.a_buf = Var(shape=output_shape, init=0)
        self.num_message_bits = Var(shape=(1,), init=num_message_bits)

    @staticmethod
    def _validate_input_shape(input_shape: ty.Tuple[int, int, int]) -> None:
        if len(input_shape) != 3:
            raise ValueError(
                f'Expected input shape to be 3 dimensional.'
                f'Found {input_shape}.'
            )

    @staticmethod
    def _validate_groups(groups: int) -> None:
        if not np.isscalar(groups):
            raise ValueError(
                f'Expected groups to be a scalar.'
                f'found {groups = }.'
            )

    @staticmethod
    def _validate_channels(in_channels: int,
                           out_channels: int,
                           groups: int) -> None:
        if in_channels % groups != 0:
            raise ValueError(
                f'Expected number of in_channels to be divisible by groups.'
                f'Found {in_channels = } and {groups = }.'
            )
        if out_channels % groups != 0:
            raise ValueError(
                f'Expected number of out_channels to be divisible by groups.'
                f'Found {out_channels = } and {groups = }.'
            )
