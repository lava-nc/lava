# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.proc.conv import utils


class MaxPooling(AbstractProcess):
    def __init__(
            self,
            *,
            shape_in: ty.Tuple[int, int, int],
            kernel_size: ty.Union[int, ty.Tuple[int, int]],
            stride: ty.Optional[ty.Union[int, ty.Tuple[int, int]]] = None,
            padding: ty.Optional[ty.Union[int, ty.Tuple[int, int]]] = (0, 0),
            **kwargs) -> None:
        super().__init__(shape_in=shape_in,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         **kwargs)

        self._validate_shape_in(shape_in)

        if stride is None:
            stride = kernel_size

        in_channels = shape_in[-1]
        out_channels = in_channels

        kernel_size = utils.make_tuple(kernel_size)
        stride = utils.make_tuple(stride)
        padding = utils.make_tuple(padding)

        self._validate_kernel_size(kernel_size)
        self._validate_stride(stride)
        self._validate_padding(padding)

        shape_out = utils.output_shape(
            shape_in, out_channels, kernel_size, stride, padding, (1, 1)
        )

        self.in_port = InPort(shape=shape_in)
        self.out_port = OutPort(shape=shape_out)
        self.kernel_size = Var(shape=(2,), init=kernel_size)
        self.padding = Var(shape=(2,), init=padding)
        self.stride = Var(shape=(2,), init=stride)

    @staticmethod
    def _validate_shape_in(shape_in: ty.Tuple[int, int, int]) -> None:
        if not len(shape_in) == 3:
            raise ValueError(f"shape_in should be 3 dimensional. {shape_in} given.")

        if shape_in[0] <= 0 or shape_in[1] <= 0:
            raise ValueError(f"Width and height of shape_in should be positive."
                             f"{shape_in} given.")

    @staticmethod
    def _validate_kernel_size(kernel_size: ty.Tuple[int, int]) -> None:
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError(f"Kernel size elements should be strictly positive."
                             f"{kernel_size=} found.")
        
    @staticmethod
    def _validate_stride(stride: ty.Tuple[int, int]) -> None:
        if stride[0] <= 0 or stride[1] <= 0:
            raise ValueError(f"Stride elements should be strictly positive."
                             f"{stride=} found.")
        
    @staticmethod
    def _validate_padding(padding: ty.Tuple[int, int]) -> None:
        if padding[0] < 0 or padding[1] < 0:
            raise ValueError(f"Padding elements should be positive."
                             f"{padding=} found.")