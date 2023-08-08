# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF
from lava.magma.core.run_configs import Loihi2HwCfg

from lava.magma.core.resources import LMT
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.c.model import CLoihiProcessModel
from lava.magma.core.model.c.type import LavaCType, LavaCDataType
from lava.magma.core.model.c.ports import CInPort, COutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

class CProcBubble(AbstractProcess):
    def __init__(self, size):
        super().__init__()
        self.iport = InPort((size,))
        self.oport = OutPort((size,))

@implements(proc=CProcBubble, protocol=LoihiProtocol)
@requires(LMT)
class CProcBubbleModel(CLoihiProcessModel):
    iport: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
    oport: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)

    @property
    def source_file_name(self) -> str:
        return "dummy.c"

class PerceptronLayer(AbstractProcess):
    def __init__(self, size, du, dv, bias_mant, bias_exp, vth) -> None:
        super().__init__(shape=(size,), du=du, dv=dv,
                         bias_mant=bias_mant, bias_exp=bias_exp, vth=vth)
        self.iport = InPort(shape=(size,))
        self.oport = OutPort(shape=(size,))

        weights_ = np.ones(size * size).reshape(size, size) * size
        dense_params_ = {'weights': weights_}
        self.dense1 = Dense(**dense_params_)

        self.lif1 = LIF(shape=(size,), du=du, dv=dv,
                       bias_mant=bias_mant, bias_exp=bias_exp, vth=vth)

        dense_params_ = {'weights': weights_+1}
        self.dense2 = Dense(**dense_params_)

        self.lif2 = LIF(shape=(size,), du=du+1, dv=dv+1,
                       bias_mant=bias_mant+1, bias_exp=bias_exp+1, vth=vth+1)

        self.iport.connect(self.dense1.s_in)
        self.dense1.a_out.connect(self.lif1.a_in)
        self.lif1.s_out.connect(self.dense2.s_in)
        self.dense2.a_out.connect(self.lif2.a_in)
        self.lif2.s_out.connect(self.oport)

class TestFoldedView(unittest.TestCase):
    def test_folded_view_mlp(self):
        size = 6
        layer1 = PerceptronLayer(size=size,
                                 du=4095,
                                 dv=1024,
                                 bias_mant=10,
                                 bias_exp=6,
                                 vth=10)

        bubble = CProcBubble(size=size)

        layer2 = PerceptronLayer(size=size,
                                 du=4095,
                                 dv=1024,
                                 bias_mant=13,
                                 bias_exp=3,
                                 vth=3)

        layer1.oport.connect(bubble.iport)
        bubble.oport.connect(layer2.iport)

        run_cfg = Loihi2HwCfg()
        compile_config = {'folded_view' : ['PerceptronLayer']}

        layer2.compile(run_cfg=run_cfg, compile_config=compile_config)

        self.assertEqual(layer1.folded_view, layer2.folded_view)
        self.assertEqual(layer1.folded_view.__name__, compile_config['folded_view'][0])

        self.assertEqual(layer1.procs.lif1.folded_view, layer1.folded_view)
        self.assertEqual(layer1.procs.lif1.folded_view_inst_id, layer1.folded_view_inst_id)
        self.assertEqual(layer1.procs.dense1.folded_view, layer1.folded_view)
        self.assertEqual(layer1.procs.dense1.folded_view_inst_id, layer1.folded_view_inst_id)
        self.assertEqual(layer1.procs.lif2.folded_view, layer1.folded_view)
        self.assertEqual(layer1.procs.lif2.folded_view_inst_id, layer1.folded_view_inst_id)
        self.assertEqual(layer1.procs.dense2.folded_view, layer1.folded_view)
        self.assertEqual(layer1.procs.dense2.folded_view_inst_id, layer1.folded_view_inst_id)

        self.assertEqual(layer2.procs.lif1.folded_view, layer2.folded_view)
        self.assertEqual(layer2.procs.lif1.folded_view_inst_id, layer2.folded_view_inst_id)
        self.assertEqual(layer2.procs.dense1.folded_view, layer2.folded_view)
        self.assertEqual(layer2.procs.dense1.folded_view_inst_id, layer2.folded_view_inst_id)
        self.assertEqual(layer2.procs.lif2.folded_view, layer2.folded_view)
        self.assertEqual(layer2.procs.dense2.folded_view, layer2.folded_view)
        self.assertEqual(layer2.procs.lif2.folded_view_inst_id, layer2.folded_view_inst_id)
        self.assertEqual(layer2.procs.dense2.folded_view_inst_id, layer2.folded_view_inst_id)

        self.assertNotEqual(layer1.folded_view_inst_id, layer2.folded_view_inst_id)

if __name__ == '__main__':
    unittest.main()
