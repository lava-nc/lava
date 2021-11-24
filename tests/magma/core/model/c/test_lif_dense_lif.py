# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig
from lava.proc.lif.process import LIF
import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.magma.core.model.c.model import AbstractCProcessModel

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class CDense(AbstractProcess):
    """Dense connections between neurons.
    Realizes the following abstract behavior:
    a_out = W * s_in
    """

    def __init__(self, **kwargs):
        # super(AbstractProcess, self).__init__(kwargs)
        # shape = kwargs.pop("shape")
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1],))
        self.a_out = OutPort(shape=(shape[0],))
        self.weights = Var(shape=shape, init=kwargs.pop("weights", 0))


class SimpleRunConfig(RunConfig):
    def __init__(self, **kwargs):
        sync_domains = kwargs.pop("sync_domains")
        super().__init__(custom_sync_domains=sync_domains)
        self.model = None
        if "model" in kwargs:
            self.model = kwargs.pop("model")

    def select(self, process, proc_models):
        if self.model is not None:
            if self.model == "sub" and isinstance(process, AbstractProcess):
                return proc_models[1]
        return proc_models[0]


class TestLifDenseLif(unittest.TestCase):
    def test_lif_dense_lif(self):
        self.lif1 = LIF()
        self.dense = CDense()
        self.lif2 = LIF()
        self.lif1.out_ports.s_out.connect(self.dense.in_ports.s_in)
        self.dense.out_ports.a_out.connect(self.lif2.in_ports.a_in)
        self.lif1.run(
            condition=RunSteps(num_steps=10),
            run_cfg=SimpleRunConfig(sync_domains=[]),
        )
        self.lif1.stop()


if __name__ == "__main__":
    unittest.main()
