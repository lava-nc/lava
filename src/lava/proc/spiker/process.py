# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy.typing as npty

import numpy as np
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class Spiker(AbstractProcess):
    """Process emitting a specified payload at a given rate.

    Parameters
    ----------
    shape : tuple(int)
        Shape of the population of process units.
    period : int
        Number of timesteps between subsequent emissions of payload.
        Note that the first spike is emitted at time step period + 1.
    payload : int
        A value to be send with every output message.
    name : str
        Name of the Process. Default is 'Process_ID', where ID is an
        integer value that is determined automatically.
    log_config : LogConfig
        Configuration options for logging.
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


class Spiker32bit(AbstractProcess):
    """Process emitting a specified payload at a given rate.
    Provides 32bit payloads, and separate payloads for each neuron.
    Other than the default Spiker process, this process actually starts spiking
    at timestep = period.

    Parameters
    ----------
    shape : tuple(int)
        Shape of the population of process units.
    period : int
        Number of timesteps between subsequent emissions of payload.
    payload : int
        A value to be send with every output message.
        Can be in [0, 2**32 - 1] if signed==False,
        or in [-2**31, 2**31 - 1] if signed==True.
    signed : bool
        True if signed payload, False otherwise.
    name : str
        Name of the Process. Default is 'Process_ID', where ID is an
        integer value that is determined automatically.
    log_config : LogConfig
        Configuration options for logging.
    """

    def __init__(self, *,
                 shape: ty.Tuple[int, ...] = (1,),
                 period: ty.Union[int, npty.ArrayLike] = 1,
                 payload: ty.Union[int, npty.ArrayLike] = 1,
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:

        signed = self._input_validation(payload)

        if np.isscalar(period):
            period = np.array([period], dtype=int)
        else:
            period = period.astype(int)
        if np.isscalar(payload):
            payload = np.array([payload])
        else:
            payload = payload.astype(int)

        super().__init__(shape=shape,
                         period=period,
                         payload=payload,
                         signed=signed,
                         name=name, log_config=log_config)
        self.s_out = OutPort(shape=shape)
        self.counter = Var(shape=shape, init=np.zeros(shape).astype(int) + 1)

    def _input_validation(self, payload) -> bool:
        payload_min = np.min(payload)
        payload_max = np.max(payload)
        signed = payload_min < 0

        if payload_min < -2 ** 31:
            raise ValueError(
                f"The payload must be >= -2**31, but the smallest value is "
                f"{payload_min}.")

        payload_max_allowed = 2 ** 31 - 1 if signed else 2 ** 32 - 1

        if payload_max > payload_max_allowed:
            raise ValueError(
                f"The payload must be <= {payload_max_allowed}, but the "
                f"largest value is  {payload_max}.")

        return signed
