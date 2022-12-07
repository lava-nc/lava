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
    shape : tuple
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
        """Validate that a given shape is of the right format (max_num_events, )

        Parameters
        ----------
        shape : tuple
            Shape to validate.
        """
        if len(shape) != 1:
            raise ValueError(f"Shape should be (n,). {shape} was given.")

        if shape[0] <= 0:
            raise ValueError(f"Max number of events should be positive. "
                             f"{shape} was given.")