# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF


class SimpleRunConfig(RunConfig):
    def __init__(self, **kwargs):
        sync_domains = kwargs.pop("sync_domains")
        self.model = None
        super().__init__(custom_sync_domains=sync_domains)
        if "model" in kwargs:
            self.model = kwargs.pop("model")

    def select(self, process, proc_models):
        if self.model is not None:
            if self.model == "sub" and isinstance(process, AbstractProcess):
                return proc_models[1]
        return proc_models[0]


class TestLifDenseLif(unittest.TestCase):
    def test_lif_dense_lif(self):
        lif1 = LIF(shape=(1,))
        dense = Dense(weights=np.eye(1))
        lif2 = LIF(shape=(1,))
        lif1.out_ports.s_out.connect(dense.in_ports.s_in)
        dense.out_ports.a_out.connect(lif2.in_ports.a_in)
        lif1.run(condition=RunSteps(num_steps=10),
                 run_cfg=SimpleRunConfig(sync_domains=[]))
        lif1.stop()
