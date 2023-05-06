# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class DemoProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        nb_runs = kwargs.pop("nb_runs")
        self.changed = Var(shape=(1,), init=np.array([True]))
        self.changed_history = Var(shape=(nb_runs,), init=np.nan)


@implements(proc=DemoProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class DemoProcessModel(PyLoihiProcessModel):
    changed: np.ndarray = LavaPyType(np.ndarray, bool)
    changed_history: np.ndarray = LavaPyType(np.ndarray, bool)

    def run_spk(self):
        self.changed_history[self.time_step - 1] = self.changed[0]
        self.changed[0] = False


class TestNonDeterminismUpdate(unittest.TestCase):
    def test_non_determinism_update(self):
        nb_runs = 10000
        demo_process = DemoProcess(nb_runs=nb_runs)
        for _ in range(nb_runs):
            demo_process.run(condition=RunSteps(num_steps=1),
                             run_cfg=Loihi1SimCfg())

            demo_process.changed.set(np.array([True]))

            # Uncomment this, test will pass
            # Comment this, test will probably fail
            # demo_process.changed.get()

        changed_history = demo_process.changed_history.get()

        demo_process.stop()

        np.testing.assert_array_equal(np.ones((nb_runs,)), changed_history)


if __name__ == '__main__':
    unittest.main()
