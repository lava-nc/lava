# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import typing as ty
import numpy as np

from lava.magma.core.decorator import implements
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, RefPort, InPort, \
    ChannelStub
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class POut(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_port = OutPort(shape=(2,))


class PIn(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_port = InPort(shape=(2, ))


class PVar(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.var = Var(shape=(2,), init=4)


class PRef(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_port = RefPort(shape=(2,))


@implements(proc=POut)
class PyProcModelPOut(PyLoihiProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)


@implements(proc=PIn)
class PyProcModelPIn(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)


@implements(proc=PRef)
class PyProcModelPRef(PyLoihiProcessModel):
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)


@implements(proc=PVar)
class PyProcModelPVar(PyLoihiProcessModel):
    var: np.ndarray = LavaPyType(np.ndarray, np.int32)


class TestChannelStub(unittest.TestCase):
    def test_connect_in_out(self):
        p_out = POut()
        p_in = PIn()
        c: ChannelStub = p_out.out_port.connect(p_in.in_port)
        configuration = {"x": "1", "y": 2}
        c.configure(configuration)

        self.assertDictEqual(p_out.out_port.config[p_in.in_port], configuration)
        self.assertDictEqual(p_in.in_port.config[p_out.out_port], configuration)

    def test_connect_in_out_backwards(self):
        p_out = POut()
        p_in = PIn()
        c: ChannelStub = p_in.in_port.connect_from(p_out.out_port)
        configuration = {"x": "1", "y": 2}
        c.configure(configuration)

        self.assertDictEqual(p_out.out_port.config[p_in.in_port], configuration)
        self.assertDictEqual(p_in.in_port.config[p_out.out_port], configuration)

    def test_connect_in_out_join(self):
        p_out1 = POut()
        p_out2 = POut()
        p_in = PIn()
        c: ty.List[ChannelStub] = \
            p_in.in_port.connect_from([p_out1.out_port, p_out2.out_port])
        c1_configuration = {"x": "1", "y": 1}
        c2_configuration = {"x": "2", "y": 2}
        c[0].configure(c1_configuration)
        c[1].configure(c2_configuration)

        self.assertDictEqual(p_out1.out_port.config[p_in.in_port],
                             c1_configuration)
        self.assertDictEqual(p_out2.out_port.config[p_in.in_port],
                             c2_configuration)

        self.assertDictEqual(p_in.in_port.config[p_out1.out_port],
                             c1_configuration)
        self.assertDictEqual(p_in.in_port.config[p_out2.out_port],
                             c2_configuration)

    def test_connect_in_out_fork(self):
        p_out = POut()
        p_in1 = PIn()
        p_in2 = PIn()
        c: ty.List[ChannelStub] = \
            p_out.out_port.connect([p_in1.in_port, p_in2.in_port])
        c1_configuration = {"x": "1", "y": 1}
        c2_configuration = {"x": "2", "y": 2}
        c[0].configure(c1_configuration)
        c[1].configure(c2_configuration)

        self.assertDictEqual(p_in1.in_port.config[p_out.out_port],
                             c1_configuration)
        self.assertDictEqual(p_in2.in_port.config[p_out.out_port],
                             c2_configuration)

        self.assertDictEqual(p_out.out_port.config[p_in1.in_port],
                             c1_configuration)
        self.assertDictEqual(p_out.out_port.config[p_in2.in_port],
                             c2_configuration)

    def test_connect_var_ref(self):
        p_ref = PRef()
        p_var = PVar()
        c: ChannelStub = p_ref.ref_port.connect_var(p_var.var)
        configuration = {"x": "1", "y": 2}
        c.configure(configuration)

        var_port = p_var.var_ports.members[0]

        self.assertDictEqual(p_ref.ref_port.config[var_port],
                             configuration)
        self.assertDictEqual(var_port.config[p_ref.ref_port],
                             configuration)


if __name__ == '__main__':
    unittest.main()
