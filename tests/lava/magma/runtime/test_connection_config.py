# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np

from lava.magma.core.decorator import implements
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, RefPort, InPort
from lava.magma.core.process.ports.connection_config import ConnectionConfig
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


class TestConnectionConfig(unittest.TestCase):
    def test_connect_in_out(self):
        NUM_PROBES = 8
        ETHERNET_PACKET_LEN = 128
        p_out = POut()
        p_in = PIn()
        config = ConnectionConfig(num_probes=NUM_PROBES,
                                  ethernet_packet_len=ETHERNET_PACKET_LEN)
        p_out.out_port.connect(p_in.in_port, config)

        self.assertEqual(p_out.out_port.connection_configs[p_in.in_port],
                         config)
        self.assertEqual(p_in.in_port.connection_configs[p_out.out_port],
                         config)
        self.assertEqual(
            p_out.out_port.connection_configs[p_in.in_port].num_probes,
            NUM_PROBES)
        self.assertEqual(
            p_out.out_port.connection_configs[p_in.in_port].ethernet_packet_len,
            ETHERNET_PACKET_LEN)

    def test_connect_in_out_backwards(self):
        NUM_PROBES = 8
        ETHERNET_PACKET_LEN = 128
        p_out = POut()
        p_in = PIn()
        config = ConnectionConfig(num_probes=NUM_PROBES,
                                  ethernet_packet_len=ETHERNET_PACKET_LEN)
        p_in.in_port.connect_from(p_out.out_port, config)

        self.assertEqual(p_out.out_port.connection_configs[p_in.in_port],
                         config)
        self.assertEqual(p_in.in_port.connection_configs[p_out.out_port],
                         config)
        self.assertEqual(
            p_out.out_port.connection_configs[p_in.in_port].num_probes,
            NUM_PROBES)
        self.assertEqual(
            p_out.out_port.connection_configs[p_in.in_port].ethernet_packet_len,
            ETHERNET_PACKET_LEN)

    def test_connect_in_out_join(self):
        NUM_PROBES1 = 8
        ETHERNET_PACKET_LEN1 = 128

        NUM_PROBES2 = 2
        ETHERNET_PACKET_LEN2 = 16

        p_out1 = POut()
        p_out2 = POut()
        p_in = PIn()

        c1_configuration = ConnectionConfig(
            num_probes=NUM_PROBES1,
            ethernet_packet_len=ETHERNET_PACKET_LEN1)
        c2_configuration = ConnectionConfig(
            num_probes=NUM_PROBES2,
            ethernet_packet_len=ETHERNET_PACKET_LEN2)

        p_in.in_port.connect_from(
            [p_out1.out_port, p_out2.out_port],
            [c1_configuration, c2_configuration])

        self.assertEqual(p_out1.out_port.connection_configs[p_in.in_port],
                         c1_configuration)
        self.assertEqual(p_out2.out_port.connection_configs[p_in.in_port],
                         c2_configuration)

        self.assertEqual(p_in.in_port.connection_configs[p_out1.out_port],
                         c1_configuration)
        self.assertEqual(p_in.in_port.connection_configs[p_out2.out_port],
                         c2_configuration)

    def test_connect_in_out_fork(self):
        NUM_PROBES1 = 8
        ETHERNET_PACKET_LEN1 = 128

        NUM_PROBES2 = 2
        ETHERNET_PACKET_LEN2 = 16

        p_out = POut()
        p_in1 = PIn()
        p_in2 = PIn()

        c1_configuration = ConnectionConfig(
            num_probes=NUM_PROBES1,
            ethernet_packet_len=ETHERNET_PACKET_LEN1)
        c2_configuration = ConnectionConfig(
            num_probes=NUM_PROBES2,
            ethernet_packet_len=ETHERNET_PACKET_LEN2)

        p_out.out_port.connect([p_in1.in_port, p_in2.in_port],
                               [c1_configuration, c2_configuration])

        self.assertEqual(p_in1.in_port.connection_configs[p_out.out_port],
                         c1_configuration)
        self.assertEqual(p_in2.in_port.connection_configs[p_out.out_port],
                         c2_configuration)

        self.assertEqual(p_out.out_port.connection_configs[p_in1.in_port],
                         c1_configuration)
        self.assertEqual(p_out.out_port.connection_configs[p_in2.in_port],
                         c2_configuration)

    def test_connect_var_ref(self):
        p_ref = PRef()
        p_var = PVar()
        configuration = ConnectionConfig()
        p_ref.ref_port.connect_var(p_var.var, configuration)

        var_port = p_var.var_ports.members[0]

        self.assertEqual(p_ref.ref_port.connection_configs[var_port],
                         configuration)
        self.assertEqual(var_port.connection_configs[p_ref.ref_port],
                         configuration)


if __name__ == '__main__':
    unittest.main()
