# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.spiker.process import Spiker


class TestSpikerModels(unittest.TestCase):
    """Tests spiker PyProcModel."""

    def test_single_spiker_counter(self):
        "Tests a single spiker for multiple time steps."
        spiker = Spiker(shape=(1,), period=5)
        counter = []
        for _ in range(20):
            spiker.run(condition=RunSteps(num_steps=1),
                       run_cfg=Loihi2SimCfg())
            counter.append(spiker.counter.get()[0])
        spiker.stop()
        expected = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0,
                               5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
                               4.0, 5.0])
        self.assertTrue(np.all(counter == expected))

    def test_multiple_spikers_counter(self):
        spiker = Spiker(shape=(2,), period=5)
        counter1 = []
        counter2 = []
        for _ in range(20):
            spiker.run(condition=RunSteps(num_steps=1),
                       run_cfg=Loihi2SimCfg())
            counter1.append(spiker.counter.get()[0])
            counter2.append(spiker.counter.get()[1])
        spiker.stop()
        expected = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0,
                               5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
                               4.0, 5.0])
        self.assertTrue(np.all(counter1 == expected))
        self.assertTrue(np.all(counter2 == expected))
