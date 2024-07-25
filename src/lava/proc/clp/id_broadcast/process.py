# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess, LogConfig


class IdBroadcast(AbstractProcess):
    """Process that sends out a graded spike with payload equal to a_in.

    Parameters
    ----------
    shape : tuple(int)
        Shape of the population of process units.
    name : str
        Name of the Process. Default is 'Process_ID', where ID is an
        integer value that is determined automatically.
    log_config : LogConfig
        Configuration options for logging.
    """

    def __init__(self, *,
                 shape: ty.Tuple[int, ...] = (1,),
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        super().__init__(shape=shape, name=name, log_config=log_config)
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.val = Var(shape=shape, init=0)
