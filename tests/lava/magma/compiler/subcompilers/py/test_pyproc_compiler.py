# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import typing as ty

from lava.magma.compiler.subcompilers.py.pyproc_compiler import PyProcCompiler
from lava.magma.compiler.builders.py_builder import PyProcessBuilder
from lava.magma.compiler.channel_map import ChannelMap, PortPair, Payload
from lava.magma.core.decorator import implements, requires
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.reduce_ops import ReduceSum
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.process.ports.ports import (
    InPort,
    OutPort,
    RefPort,
    VarPort,
)
from lava.magma.core.model.py.ports import (
    PyInPort,
    PyOutPort,
    PyRefPort,
    PyVarPort,
)
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.process.ports.ports import create_port_id


# A minimal process (A) with an InPort, OutPort and RefPort.
class ProcA(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use ReduceOp to allow for multiple input connections.
        self.inp = InPort(shape=(1,), reduce_op=ReduceSum)
        self.out = OutPort(shape=(1,))
        self.ref = RefPort(shape=(10,))


# Another minimal process (B) with a Var and an InPort, OutPort and VarPort.
class ProcB(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use ReduceOp to allow for multiple input connections.
        self.inp = InPort(shape=(1,), reduce_op=ReduceSum)
        self.out = OutPort(shape=(1,))
        self.some_var = Var((10,), init=10)
        self.var_port = VarPort(self.some_var)


class MockRuntimeService:
    __name__ = "MockRuntimeService"


# Define minimal Protocol to be implemented.
class ProtocolA(AbstractSyncProtocol):
    @property
    def runtime_service(self):
        return {CPU: MockRuntimeService()}


# Define minimal Protocol to be implemented.
class ProtocolB(AbstractSyncProtocol):
    @property
    def runtime_service(self):
        return {CPU: MockRuntimeService()}


# A minimal PyProcModel implementing ProcA.
@implements(proc=ProcA, protocol=ProtocolA)
@requires(CPU)
class PyProcModelA(AbstractPyProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    ref: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)

    def run(self):
        pass


# A minimal PyProcModel implementing ProcB.
@implements(ProcB, protocol=ProtocolB)
@requires(CPU)
class PyProcModelB(AbstractPyProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    some_var: int = LavaPyType(int, int)
    var_port: PyVarPort = LavaPyType(PyVarPort.VEC_DENSE, int)

    def run(self):
        pass


class TestPyProcCompiler(unittest.TestCase):
    def setUp(self) -> None:
        """Setting compiler config for PyProcCompiler."""
        self.compile_config = {"pypy_channel_size": 64}

    def test_compile_py_proc_models(self):
        """Checks compilation of ProcessModels which (in this example) only
        generates PyProcBuilders for each Process and ProcessModel."""

        # Normally, the overall Lava Compiler would have already generated
        # ProcGroups and would create a SubCompiler for each
        # ProcGroup. We create a ProcGroup here manually.
        p1, p2, p3 = ProcA(), ProcA(), ProcB()
        p1._model_class = PyProcModelA
        p2._model_class = PyProcModelA
        p3._model_class = PyProcModelB
        proc_group = [p1, p2, p3]

        # Compiling this ProcGroup, returning initialized PyProcBuilders.
        channel_map = ChannelMap()
        py_proc_compiler = PyProcCompiler(
            proc_group=proc_group, compile_config=self.compile_config
        )
        py_proc_compiler.compile(channel_map)
        builders, channel_map = py_proc_compiler.get_builders(channel_map)

        # There should be three PyProcessBuilders...
        self.assertEqual(len(builders), 3)
        for builder in builders.values():
            self.assertIsInstance(builder, PyProcessBuilder)
        # ... one for each Process.
        b1 = ty.cast(PyProcessBuilder, builders[p1])
        b2 = ty.cast(PyProcessBuilder, builders[p2])
        b3 = ty.cast(PyProcessBuilder, builders[p3])

        # ProcA builders only have PortInitializers for 'inp' and 'out'...
        for procA_b in [b1, b2]:
            self.assertEqual(len(procA_b.vars), 0)
            self.assertEqual(procA_b.py_ports["inp"].name, "inp")
            self.assertEqual(procA_b.py_ports["inp"].shape, (1,))
            self.assertEqual(procA_b.py_ports["out"].name, "out")

        # ... while ProcB has also a VarInitializer.
        self.assertEqual(len(b3.vars), 1)
        self.assertEqual(b3.vars["some_var"].name, "some_var")
        self.assertEqual(b3.vars["some_var"].value, 10)
        self.assertEqual(b3.py_ports["inp"].name, "inp")

    def test_compile_channel_map(self):
        """Checks whether the channel map is updated correctly by the
        PyProcCompiler."""

        # Normally, the overall Lava Compiler would have already generated
        # ProcGroups and would create a SubCompiler for each
        # ProcGroup. We create a ProcGroup here manually.
        p1, p2, p3 = ProcA(), ProcA(), ProcB()
        p1.out.connect(p2.inp)
        p1.out.connect(p3.inp)
        p1.ref.connect(p3.var_port)
        p1._model_class = PyProcModelA
        p2._model_class = PyProcModelA
        p3._model_class = PyProcModelB
        proc_group = [p1, p2, p3]

        # Create more Processes that are in a different ProcGroup but
        # connected to this ProcGroup.
        p4, p5 = ProcA(), ProcB()
        p4.out.connect(p1.inp)
        p3.out.connect(p5.inp)

        # Compile this ProcGroup.
        py_proc_compiler = PyProcCompiler(
            proc_group=proc_group, compile_config=self.compile_config
        )
        channel_map = py_proc_compiler.compile(ChannelMap())

        # There should be five entries in the channel map...
        self.assertEqual(len(channel_map), 5)
        # ...one for each connection between Ports...
        pl = Payload(multiplicity=1)
        self.assertEqual(channel_map[PortPair(src=p1.out, dst=p2.inp)], pl)
        self.assertEqual(channel_map[PortPair(src=p1.out, dst=p3.inp)], pl)
        self.assertEqual(channel_map[PortPair(src=p1.ref, dst=p3.var_port)], pl)
        # ...including connections to Ports of Processes that are in other
        # ProcGroups.
        self.assertEqual(channel_map[PortPair(src=p4.out, dst=p1.inp)], pl)
        self.assertEqual(channel_map[PortPair(src=p3.out, dst=p5.inp)], pl)

        # Check that the channel map is not changed when compile() is called
        # a second time.
        cm_first_iteration = channel_map.copy()
        py_proc_compiler.compile(ChannelMap())
        self.assertEqual(channel_map, cm_first_iteration)

    def test_compile_py_proc_models_with_virtual_ports(self):
        """Checks compilation of ProcessModels when Processes are connected
        via virtual ports."""

        # Normally, the overall Lava Compiler would have already generated
        # ProcGroups and would create a SubCompiler for each
        # ProcGroup. We create a ProcGroup here manually.
        p1, p2, p3 = ProcA(), ProcA(), ProcB()
        p1._model_class = PyProcModelA
        p2._model_class = PyProcModelA
        p3._model_class = PyProcModelB
        p1.out.flatten().connect(p3.inp)
        p2.ref.flatten().connect(p3.var_port)
        proc_group = [p1, p2, p3]
        channel_map = ChannelMap()

        # Compiling this ProcGroup, returning initialized PyProcBuilders.
        py_proc_compiler = PyProcCompiler(
            proc_group=proc_group, compile_config=self.compile_config
        )
        py_proc_compiler.compile(channel_map)
        builders, channel_map = py_proc_compiler.get_builders(channel_map)

        # Get the individual builders from the dictionary.
        b2 = ty.cast(PyProcessBuilder, builders[p2])
        b3 = ty.cast(PyProcessBuilder, builders[p3])

        # Check whether the transformation functions are registered in the
        # PortInitializers.
        self.assertEqual(
            list(b3.py_ports["inp"].transform_funcs.keys()),
            [create_port_id(p1.id, p1.out.name)],
        )
        self.assertEqual(
            list(b3.var_ports["var_port"].transform_funcs.keys()),
            [create_port_id(p2.id, p2.ref.name)],
        )
        self.assertEqual(
            list(b2.ref_ports["ref"].transform_funcs.keys()),
            [create_port_id(p3.id, p3.var_port.name)],
        )


if __name__ == "__main__":
    unittest.main()
