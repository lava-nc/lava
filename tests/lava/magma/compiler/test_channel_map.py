# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.compiler.channel_map import ChannelMap
from lava.magma.compiler.channel_map import PortPair, Payload
from lava.magma.compiler.subcompilers.channel_builders_factory import (
    ChannelBuildersFactory,
)
from lava.magma.compiler.utils import PortInitializer
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.process.ports.reduce_ops import ReduceSum
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocol import AbstractSyncProtocol


# A minimal process (A) with an InPort, OutPort and RefPort
class ProcA(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use ReduceOp to allow for multiple input connections
        self.inp = InPort(shape=(1,), reduce_op=ReduceSum)
        self.out = OutPort(shape=(1,))
        self.ref = RefPort(shape=(10,))


class MockRuntimeService:
    __name__ = "MockRuntimeService"


# Define minimal Protocol to be implemented
class ProtocolA(AbstractSyncProtocol):
    @property
    def runtime_service(self):
        return {CPU: MockRuntimeService()}


# A minimal PyProcModel implementing ProcA
@implements(proc=ProcA, protocol=ProtocolA)
@requires(CPU)
class PyProcModelA(AbstractPyProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    ref: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)

    def run(self):
        pass


class TestChannelMap(unittest.TestCase):
    def setUp(self) -> None:
        self.channel_map = ChannelMap()
        self.inport1 = InPort(shape=(10, 2))
        self.inport2 = InPort(shape=(10, 2))
        self.outport1 = OutPort(shape=(10, 2))
        self.outport2 = OutPort(shape=(10, 2))
        self.port_pair1 = PortPair(src=self.outport1, dst=self.inport1)
        self.port_pair2 = PortPair(src=self.outport2, dst=self.inport2)

    def test_obj_creation(self) -> None:
        self.assertIsInstance(self.channel_map, ChannelMap)

    def test_set_multiplicity(self) -> None:
        payload = Payload(multiplicity=5)
        self.channel_map[self.port_pair1] = payload
        self.assertEqual(self.channel_map[self.port_pair1], payload)

    def test_set_tiling(self) -> None:
        payload = Payload(multiplicity=2, tiling=(2, 2, 5))
        self.channel_map[self.port_pair1] = payload
        self.assertEqual(self.channel_map[self.port_pair1], payload)

    def test_wrong_key_type_causes_type_error(self) -> None:
        payload = Payload(multiplicity=5)
        with self.assertRaises(TypeError):
            self.channel_map["string"] = payload

    def test_wrong_value_type_causes_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self.channel_map[self.port_pair1] = "foo"

    def test_equal_channel_maps_comparison(self) -> None:
        cm1 = ChannelMap()
        cm1[self.port_pair1] = Payload(multiplicity=2)
        cm1[self.port_pair2] = Payload(multiplicity=4)
        cm2 = ChannelMap()
        cm2[self.port_pair1] = Payload(multiplicity=2)
        cm2[self.port_pair2] = Payload(multiplicity=4)
        self.assertEqual(cm1, cm2)

    def test_different_channel_maps_comparison(self) -> None:
        cm1 = ChannelMap()
        cm1[self.port_pair1] = Payload(multiplicity=2)
        cm1[self.port_pair2] = Payload(multiplicity=4)
        cm2 = ChannelMap()
        cm2[self.port_pair1] = Payload(multiplicity=2)
        cm2[self.port_pair2] = Payload(multiplicity=3)
        self.assertTrue(cm1 != cm2)

    def test_set_port_initializer(self):
        process = ProcA()
        process._model_class = PyProcModelA
        port = process.inp
        port_initializer = PortInitializer(
            port.name,
            port.shape,
            ChannelBuildersFactory.get_port_dtype(port),
            port.__class__.__name__,
            64,
            port.get_incoming_transform_funcs(),
        )
        self.channel_map.set_port_initializer(port, port_initializer)
        self.assertEqual(
            port_initializer, self.channel_map.get_port_initializer(port)
        )

    def test_set_port_initializer_twice_raises_exception(self):
        process = ProcA()
        process._model_class = PyProcModelA
        port = process.inp
        port_initializer = PortInitializer(
            port.name,
            port.shape,
            ChannelBuildersFactory.get_port_dtype(port),
            port.__class__.__name__,
            64,
            port.get_incoming_transform_funcs(),
        )
        self.channel_map.set_port_initializer(port, port_initializer)
        with self.assertRaises(AssertionError):
            self.channel_map.set_port_initializer(port, port_initializer)


if __name__ == "__main__":
    unittest.main()
