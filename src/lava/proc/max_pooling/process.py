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
        padding = utils.make_tuple(padding)
        stride = utils.make_tuple(stride)

        shape_out = utils.output_shape(
            shape_in, out_channels, kernel_size, stride, padding, (1, 1)
        )

        self.in_port = InPort(shape=shape_in)
        self.out_port = OutPort(shape=shape_out)
        self.kernel_size = Var(shape=(2,), init=kernel_size)
        self.padding = Var(shape=(2,), init=padding)
        self.stride = Var(shape=(2,), init=stride)

    @staticmethod
    def _validate_shape_in(shape_in):
        if not (len(shape_in) == 2 or len(shape_in) == 3):
            raise ValueError(f"shape_in should be 2 or 3 dimensional. "
                             f"{shape_in} given.")

        if not isinstance(shape_in[0], int) or not isinstance(shape_in[1], int):
            raise ValueError(f"Width and height of shape_in should be integers."
                             f"{shape_in} given.")
        if len(shape_in) == 3:
            if shape_in[2] != 2:
                raise ValueError(f"Third dimension of shape_in should be "
                                 f"equal to 2. "
                                 f"{shape_in} given.")

        if shape_in[0] <= 0 or shape_in[1] <= 0:
            raise ValueError(f"Width and height of shape_in should be positive."
                             f"{shape_in} given.")

        return shape_in
