# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.magma.core.run_configs import Loihi2HwCfg

class PerceptronLayer(AbstractProcess):
    def __init__(self, size, du, dv, bias_mant, bias_exp, vth) -> None:
        super().__init__(shape=(size,), du=du, dv=dv,
                         bias_mant=bias_mant, bias_exp=bias_exp, vth=vth)
        self.iport = InPort(shape=(size,))
        self.oport = OutPort(shape=(size,))

        self.lif = LIF(shape=(size,), du=du, dv=dv,
                       bias_mant=bias_mant, bias_exp=bias_exp, vth=vth)

        weights_ = np.ones(size * size).reshape(size, size) * size
        dense_params_ = {'weights': weights_}
        self.dense = Dense(**dense_params_)

        self.iport.connect(self.dense.s_in)
        self.dense.a_out.connect(self.lif.a_in)
        self.lif.s_out.connect(self.oport)

class TestfoldedView(unittest.TestCase):
    def test_folded_view_mlp(self):
        size = 10
        layer1 = PerceptronLayer(size=size,
                                 du=4095,
                                 dv=1024,
                                 bias_mant=10 + np.arange(10),
                                 bias_exp=6,
                                 vth=10)

        layer2 = PerceptronLayer(size=size,
                                 du=4095,
                                 dv=1024,
                                 bias_mant=10 + np.arange(10),
                                 bias_exp=3,
                                 vth=10)

        layer1.oport.connect(layer2.iport)

        run_cfg = Loihi2HwCfg()
        compile_config = {'folded_view' : ['PerceptronLayer']}

        layer2.compile(run_cfg=run_cfg, compile_config=compile_config)

        self.assertEqual(layer1.folded_view, layer2.folded_view)
        self.assertEqual(layer1.folded_view.__name__, compile_config['folded_view'][0])

        self.assertEqual(layer1.procs.lif.folded_view, layer1.folded_view)
        self.assertEqual(layer1.procs.dense.folded_view, layer1.folded_view)
        self.assertEqual(layer1.procs.lif.folded_view_inst_id, layer1.folded_view_inst_id)
        self.assertEqual(layer1.procs.dense.folded_view_inst_id, layer1.folded_view_inst_id)

        self.assertEqual(layer2.procs.lif.folded_view, layer2.folded_view)
        self.assertEqual(layer2.procs.dense.folded_view, layer2.folded_view)
        self.assertEqual(layer2.procs.lif.folded_view_inst_id, layer2.folded_view_inst_id)
        self.assertEqual(layer2.procs.dense.folded_view_inst_id, layer2.folded_view_inst_id)

        self.assertNotEqual(layer1.folded_view_inst_id, layer2.folded_view_inst_id)

if __name__ == '__main__':
    unittest.main()
