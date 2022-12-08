# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort


class BinaryToUnaryPolarity(AbstractProcess):
    """Process that transforms event-based data with binary
    polarity (0 for negative events, 1 for positive events) to unary
    polarity (1 for negative and positive events).

    Parameters
    ----------
    shape : tuple(int)
        Shape of InPort and OutPort.
    """
    def __init__(self,
                 *,
                 shape: ty.Tuple[int],
                 **kwargs) -> None:
        super().__init__(shape=shape,
                         **kwargs)

        self._validate_shape(shape)

        self.in_port = InPort(shape=shape)
        self.out_port = OutPort(shape=shape)

    @staticmethod
    def _validate_shape(shape: ty.Tuple[int]) -> None:
        """Validate that a given shape is of the form (max_num_events,) where
        max_num_events is positive.

        Parameters
        ----------
        shape : tuple(int)
            Shape to validate.
        """
        if len(shape) != 1:
            raise ValueError(f"Expected shape to be of the form "
                             f"(max_num_events,); got {shape=}.")

        if shape[0] <= 0:
            raise ValueError(f"Expected max number of events "
                             f"(first element of shape) to be positive; "
                             f"got {shape=}.")
