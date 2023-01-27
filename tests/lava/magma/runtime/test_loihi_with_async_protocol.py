# INTEL CONFIDENTIAL
# Copyright © 2022 Intel Corporation.

# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.

import unittest
import numpy as np
from lava.magma.core.decorator import requires, implements
from lava.magma.core.model.py.model import PyLoihiProcessModel, \
    PyAsyncProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

"""
This test checks if Process with Loihi Protocol works properly with
process with Async Protocol.
"""

class AProcess1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.in_port = InPort(shape=shape)

class AProcess2(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.out_port = OutPort(shape=shape)

@implements(proc=AProcess1, protocol=AsyncProtocol)
@requires(CPU)
class A1ProcessModel(PyAsyncProcessModel):
    in_port = LavaPyType(PyInPort.VEC_DENSE, int)
    def run_async(self):
        count = 1
        while True:
            val = self.in_port.recv()
            exp = (np.ones(
                    shape=self.in_port.shape)*count).tolist()
            if val.tolist() != exp:
                raise ValueError(f"Wrong value of val : {val.tolist()} {exp}")
            count += 1
            if count == 11:
                return

@implements(proc=AProcess2, protocol=AsyncProtocol)
@requires(CPU)
class A2ProcessModel(PyAsyncProcessModel):
    out_port = LavaPyType(PyOutPort.VEC_DENSE, int)
    def run_async(self):
        count = 1
        while True:
            self.out_port.send(np.ones(shape=self.out_port.shape)*count)
            count += 1
            if count == 11:
                return

class LProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs["shape"]
        self.var = Var(shape=shape, init=0)
        self.in_port = InPort(shape=shape)
        self.out_port = OutPort(shape=shape)

@implements(proc=LProcess, protocol=LoihiProtocol)
@requires(CPU)
class LProcessModel(PyLoihiProcessModel):
    in_port = LavaPyType(PyInPort.VEC_DENSE, int)
    out_port = LavaPyType(PyOutPort.VEC_DENSE, int)
    var = LavaPyType(np.ndarray, np.int32)

    def run_spk(self):
        var = self.in_port.recv()
        self.out_port.send(var)

class TestProcess(unittest.TestCase):
    def test_async_with_loihi_protocol(self):
        """
        Test is to send the data to A1 from A2 via LP.
        A2 -- > LP --> A1
        """
        shape = (2,)
        lp = LProcess(shape=shape)
        a1 = AProcess1(shape=shape)
        a2 = AProcess2(shape=shape)
        lp.in_port.connect_from(a2.out_port)
        a1.in_port.connect_from(lp.out_port)
        lp.run(condition=RunSteps(num_steps=10), run_cfg=Loihi2SimCfg())
        lp.stop()