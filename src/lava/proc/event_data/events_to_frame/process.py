# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort


class EventsToFrame(AbstractProcess):
    def __init__(self,
                 *,
                 shape_in: ty.Tuple[int],
                 shape_out: ty.Tuple[int, int, int],
                 **kwargs) -> None:
        super().__init__(shape_in=shape_in,
                         shape_out=shape_out,
                         **kwargs)

        self._validate_shape_in(shape_in)
        self._validate_shape_out(shape_out)

        self.in_port = InPort(shape=shape_in)
        self.out_port = OutPort(shape=shape_out)

    # TODO: Re-write error messages
    @staticmethod
    def _validate_shape_in(shape_in: ty.Tuple[int]) -> None:
        if len(shape_in) != 1:
            raise ValueError(f"Shape of the InPort should be (n,). "
                             f"{shape_in} was given.")

        if shape_in[0] <= 0:
            raise ValueError(f"Width of shape_in should be positive. {shape_in} given.")

    # TODO: Re-write error messages
    @staticmethod
    def _validate_shape_out(shape_out: ty.Tuple[int, int, int]) -> None:
        if not len(shape_out) == 3:
            raise ValueError(f"shape_out should be 3 dimensional. {shape_out} given.")

        if not (shape_out[2] == 1 or shape_out[2] == 2):
            raise ValueError(f"Depth of the shape_out argument should be an integer and equal to 2. "
                             f"{shape_out} given.")

        if shape_out[0] <= 0 or shape_out[1] <= 0:
            raise ValueError(f"Width and height of the shape_out argument should be positive. "
                             f"{shape_out} given.")
