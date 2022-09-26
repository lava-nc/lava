# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy as np
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class Spiker(AbstractProcess):
    """Process emitting a specified payload at a given rate.

    Parameters
    ----------
    shape: Shape of the population of process units.
    period: Number of timesteps between subsequent emissions of payload.
    payload: A value to be send with every output message.
    name: Name of the Process. Default is 'Process_ID', where ID is an
    integer value that is determined automatically.
    log_config: Configuration options for logging.
    """

    def __init__(self, *,
                 shape: ty.Tuple[int, ...] = (1,),
                 period: int = 10,
                 payload: int = 1,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(shape=shape, name=name, log_config=log_config)
        self.s_out = OutPort(shape=shape)
        self.rate = Var(shape=shape, init=period)
        self.counter = Var(shape=shape, init=np.zeros(shape).astype(int))
        self.payload = Var(shape=shape, init=payload)
