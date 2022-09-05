# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class PoissonRateCodeSpikeGen(AbstractProcess):
    def __init__(self, **kwargs: ty.Union[ty.Tuple[int, ...],
                                          float,
                                          int]) -> None:
        super().__init__(**kwargs)

        shape = kwargs.pop("shape")
        seed = kwargs.pop("seed", None)

        # self.proc_params["shape"] = shape
        # self.proc_params["seed"] = seed

        self.rates = Var(shape=shape, init=np.zeros(shape))

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)


@implements(proc=PoissonRateCodeSpikeGen, protocol=LoihiProtocol)
@requires(CPU)
class PoissonRateCodeSpikeGenProcessModel(PyLoihiProcessModel):
    rates: np.ndarray = LavaPyType(np.ndarray, float)

    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool)

    def __init__(self, proc_params):
        super(PoissonRateCodeSpikeGenProcessModel, self).__init__(proc_params)
        self.shape = self.proc_params["shape"]
        seed = self.proc_params["seed"]

        self.rng = np.random.default_rng(seed=seed)

    def _generate_spikes(self) -> np.ndarray:
        spikes = (self.rng.random(self.shape) < self.rates)

        return spikes

    def run_spk(self) -> None:
        # Receive pattern from PyInPort
        pattern = self.a_in.recv()

        # If the received pattern is not the null_pattern ...
        if not np.isnan(pattern).any():
            self.rates = np.clip(pattern, 0, 1)

        # Generate spike at every time step ...
        spikes = self._generate_spikes()
        # ... and send them through the PyOutPort
        self.s_out.send(spikes)
