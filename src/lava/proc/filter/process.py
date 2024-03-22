# Copyright (C) 2022-24 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty


from lava.magma.core.process.process import LogConfig, AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.neuron import LearningNeuronProcess

class ExpFilter(AbstractProcess):
    """Exponential Filter Process.

    dynamics abstracts to:
    v[t] = v[t-1] * (1-dv) + i[t]

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    value : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    dv : float, optional
        Inverse of decay time-constant for decay. Currently, only a
        single decay can be set for the entire population of neurons.

    Example
    -------
    >>> expFilter = ExpFilter(shape=(200, 15), dv=0.3)
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        value: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        tau: ty.Optional[float] = 0,
        state_exp: int = 0,
        num_message_bits: int = 16,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            shape=shape,
            value=value,
            tau=tau,
            num_message_bits=num_message_bits,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.value = Var(shape=shape, init=value)
        self.tau = Var(shape=(1,), init=tau)
        self.state_exp = Var(shape=(1,), init=state_exp)





