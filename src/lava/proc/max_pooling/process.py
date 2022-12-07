# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.proc.conv import utils


class MaxPooling(AbstractProcess):
    """Process that applies the max-pooling operation on incoming data.

    Parameters
    ----------
    shape_in : tuple(int, int, int)
        Shape of InPort.
    kernel_size : int or tuple(int, int)
        Size of the max-pooling kernel.
    stride : int or tuple(int, int), optional
        Stride size. Default is None.
        If not given, use kernel_size as max-pooling stride.
    padding : int or tuple(int, int), optional
        Padding size. Default is 0.
    """
    def __init__(
            self,
            *,
            shape_in: ty.Tuple[int, int, int],
            kernel_size: ty.Union[int, ty.Tuple[int, int]],
            stride: ty.Optional[ty.Union[int, ty.Tuple[int, int]]] = None,
            padding: ty.Optional[ty.Union[int, ty.Tuple[int, int]]] = 0,
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
        """Validate that a given shape is of the form (W, H, C) where W and H
        are strictly positive.

        Parameters
        ----------
        shape_in : tuple
            Shape to validate.
        """
        if not len(shape_in) == 3:
            raise ValueError(f"Expected shape_in to be 3D. "
                             f"Found {shape_in=}.")

        if shape_in[0] <= 0 or shape_in[1] <= 0:
            raise ValueError(f"Expected width and height "
                             f"(first and second elements of shape_in) to be "
                             f"strictly positive. "
                             f"Found {shape_in=}.")

    @staticmethod
    def _validate_kernel_size(kernel_size: ty.Tuple[int, int]) -> None:
        """Validate that a given kernel size (W, H) has W and H strictly
        positive.

        Parameters
        ----------
        kernel_size : tuple
            Kernel size to validate.
        """
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError(f"Expected kernel_size elements to be strictly "
                             f"positive. "
                             f"Found {kernel_size=}.")
        
    @staticmethod
    def _validate_stride(stride: ty.Tuple[int, int]) -> None:
        """Validate that a given stride (W, H) has W and H strictly
        positive.

        Parameters
        ----------
        stride : tuple
            Stride to validate.
        """
        if stride[0] <= 0 or stride[1] <= 0:
            raise ValueError(f"Expected stride elements to be strictly "
                             f"positive. "
                             f"Found {stride=}.")
        
    @staticmethod
    def _validate_padding(padding: ty.Tuple[int, int]) -> None:
        """Validate that a given padding (W, H) has W and H strictly
        positive.

        Parameters
        ----------
        padding : tuple
            Padding to validate.
        """
        if padding[0] < 0 or padding[1] < 0:
            raise ValueError(f"Expected padding elements to be positive. "
                             f"Found {padding=} .")
