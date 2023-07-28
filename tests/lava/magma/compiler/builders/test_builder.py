# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import unittest
import numpy as np

from lava.magma.compiler.builders.channel_builder import ChannelBuilderMp
from lava.magma.compiler.builders.py_builder import PyProcessBuilder
from lava.magma.compiler.channels.interfaces import Channel, ChannelType, \
    AbstractCspPort
from lava.magma.compiler.channels.pypychannel import (
    PyPyChannel,
    CspSendPort,
    CspRecvPort,
)
from lava.magma.compiler.utils import VarInitializer, PortInitializer, \
    VarPortInitializer
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort, \
    PyVarPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort, \
    VarPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.runtime.message_infrastructure.shared_memory_manager import (
    SharedMemoryManager,
)
from lava.magma.compiler.channels.watchdog import NoOPWatchdogManager


class MockMessageInterface:
    def __init__(self, smm):
        self.smm = smm

    def channel_class(self, channel_type: ChannelType) -> ty.Type:
        return PyPyChannel


class TestChannelBuilder(unittest.TestCase):
    def test_channel_builder(self):
        """Tests Channel Builder creation"""
        smm = SharedMemoryManager()
        try:
            port_initializer: PortInitializer = PortInitializer(
                name="mock", shape=(1, 2), d_type=np.int32,
                port_type='DOESNOTMATTER', size=64)
            channel_builder: ChannelBuilderMp = ChannelBuilderMp(
                channel_type=ChannelType.PyPy,
                src_port_initializer=port_initializer,
                dst_port_initializer=port_initializer,
                src_process=None,
                dst_process=None,
            )

            smm.start()
            mock = MockMessageInterface(smm)
            channel: Channel = channel_builder.build(mock,
                                                     NoOPWatchdogManager())
            assert isinstance(channel, PyPyChannel)
            assert isinstance(channel.src_port, CspSendPort)
            assert isinstance(channel.dst_port, CspRecvPort)

            channel.src_port.start()
            channel.dst_port.start()

            expected_data = np.array([[1, 2]])
            channel.src_port.send(data=expected_data)
            data = channel.dst_port.recv()
            assert np.array_equal(data, expected_data)

            channel.src_port.join()
            channel.dst_port.join()

        finally:
            smm.shutdown()


# A test Process with a variety of Ports and Vars of different shapes,
# with and without initial values that may require broadcasting or not
class Proc(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_port = InPort((2, 1))
        self.v1_scalar = Var((1,))
        self.v2_scalar_init = Var((1,), init=2)
        self.v3_tensor_broadcast = Var((2, 3), init=10)
        self.v4_tensor = Var((3, 2), init=[[1, 2], [3, 4], [5, 6]])
        self.out_port = OutPort((3, 2))


# A test PyProcessModel with corresponding LavaPyTypes for each Proc Port or Var
# Vars and Ports should have type annotations such that linter does not throw
# warnings in run(..) method because it otherwise assumes the type of the
# instance variables (created by Compiler) for every class variable is a
# LavaPyType.
@implements(proc=Proc)
@requires(CPU)
class ProcModel(AbstractPyProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, 8)
    v1_scalar: int = LavaPyType(int, int, 27)
    v2_scalar_init: int = LavaPyType(int, int, 27)
    v3_tensor_broadcast: np.ndarray = LavaPyType(np.ndarray, np.int32, 6)
    v4_tensor = LavaPyType(np.ndarray, np.int32, 6)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, 8)

    def add_ports_for_polling(self):
        pass

    def run(self):
        """Every PyProcModel must implement a run(..) method. Here we perform
        just some fake computation to demonstrate initialized Vars can be used.
        """
        return self.v1_scalar + 1


# A fake CspPort just to test ProcBuilder
class FakeCspPort(AbstractCspPort):
    def __init__(self, name="mock"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return (1, 2)

    @property
    def d_type(self) -> np.dtype:
        return np.int32.dtype

    @property
    def size(self) -> int:
        return 32

    def start(self):
        pass

    def join(self):
        pass


# Another Process for LavaPyType validation
class ProcForLavaPyType(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.port = InPort((1,))


# A correct ProcessModel
@implements(proc=ProcForLavaPyType)
@requires(CPU)
class ProcModelForLavaPyType0(AbstractPyProcessModel):
    port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    def add_ports_for_polling(self):
        pass


# A wrong ProcessModel with completely wrong type
@implements(proc=ProcForLavaPyType)
@requires(CPU)
class ProcModelForLavaPyType1(AbstractPyProcessModel):
    port: PyInPort = LavaPyType(123, int)  # type: ignore

    def add_ports_for_polling(self):
        pass


# A wrong ProcessModel with wrong sub type
@implements(proc=ProcForLavaPyType)
@requires(CPU)
class ProcModelForLavaPyType2(AbstractPyProcessModel):
    port: PyInPort = LavaPyType(PyInPort, int)

    def add_ports_for_polling(self):
        pass


# A wrong ProcessModel with wrong port type
@implements(proc=ProcForLavaPyType)
@requires(CPU)
class ProcModelForLavaPyType3(AbstractPyProcessModel):
    port: PyInPort = LavaPyType(PyOutPort, int)

    def add_ports_for_polling(self):
        pass


# A minimal process to test RefPorts and VarPorts
class ProcRefVar(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref = RefPort(shape=(3,))
        self.var = Var(shape=(3,), init=4)
        self.var_port = VarPort(self.var)


# A minimal PyProcModel implementing ProcRefVar
@implements(proc=ProcRefVar)
@requires(CPU)
class PyProcModelRefVar(AbstractPyProcessModel):
    ref: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)
    var: np.ndarray = LavaPyType(np.ndarray, np.int32)
    var_port: PyVarPort = LavaPyType(PyVarPort.VEC_DENSE, int)

    def add_ports_for_polling(self):
        pass


class TestPyProcessBuilder(unittest.TestCase):
    """ProcessModels are not not created directly but through a corresponding
    PyProcessBuilder. Therefore, we test both classes together."""

    def test_constructor(self):
        """Checks PyProcessBuilder can be constructed."""

        b = PyProcessBuilder(ProcModel, 0)

        self.assertIsInstance(b, PyProcessBuilder)

    def test_set_variables_and_ports(self):
        """Check variables and ports can be set."""

        # Create a new ProcBuilder
        b = PyProcessBuilder(ProcModel, 0)

        # Create Process for which we want to build PyProcModel
        proc = Proc()

        # Normally, the Compiler would create VarInitializers for every
        # Process Var holding only its name, shape and initial value
        v = [VarInitializer(v.name, v.shape, v.init, v.id) for v in proc.vars]
        # Similarly, the Compiler would create PortInitializers from all
        # ports holding only its name and shape
        ports = list(proc.in_ports) + list(proc.out_ports)
        py_ports = [
            PortInitializer(
                pt.name, pt.shape, getattr(ProcModel, pt.name).d_type,
                pt.__class__.__name__, 32
            )
            for pt in ports
        ]
        # Later, the Runtime, would normally create CspPorts that implements
        # the actual message passing via channels between PyPorts. Here we
        # just create some fake CspPorts for each PyPort.
        csp_ports = []
        for i, py_port in enumerate(py_ports):
            csp_port = FakeCspPort(py_port.name)
            csp_ports.append(csp_port)
            b.add_csp_port_mapping(f"id{i}", csp_port)

        # During compilation, the Compiler creates and then sets
        # VarInitializers and PyPortInitializers
        b.set_variables(v)
        b.set_py_ports(py_ports)
        # The Runtime sets CspPorts
        b.set_csp_ports(csp_ports)

        # All the objects are converted into dictionaries to retrieve them by
        # name
        self.assertEqual(list(b.vars.values()), v)
        self.assertEqual(list(b.py_ports.values()), py_ports)
        self.assertEqual(list(v for vv in b.csp_ports.values()
                              for v in vv), csp_ports)
        self.assertEqual(b.vars["v1_scalar"], v[0])
        self.assertEqual(b.py_ports["in_port"], py_ports[0])
        self.assertEqual(b.csp_ports["out_port"], [csp_ports[1]])
        self.assertEqual(b._csp_port_map["in_port"], {"id0": csp_ports[0]})
        self.assertEqual(b._csp_port_map["out_port"], {"id1": csp_ports[1]})

    def test_setting_non_existing_var(self):
        """Checks that setting Var not defined in ProcModel fails. Same will
        apply for Ports"""

        # Lets create a ProcBuilder and Proc
        b = PyProcessBuilder(ProcModel, 0)
        proc = Proc()

        # Also generate list of VarInitializers from lava.proc Vars...
        v = [VarInitializer(v.name, v.shape, v.init, v.id) for v in proc.vars]
        # ...but let's pretend Proc would define another Var
        v.append(VarInitializer("AnotherVar", (1, 2, 3), 100, 0))

        # This fails because there exists no LavaPyType for 'AnotherVar' in
        # ProcModel
        with self.assertRaises(AssertionError):
            b.set_variables(v)

    def test_check_all_vars_and_ports_set(self):
        """Checks that all Vars and Ports for which LavaPyType exists must
        be set."""

        # Lets create a ProcBuilder and Proc with Var- and PortInitializers
        b = PyProcessBuilder(ProcModel, 0)
        proc = Proc()
        v = [VarInitializer(v.name, v.shape, v.init, v.id) for v in proc.vars]
        ports = list(proc.in_ports) + list(proc.out_ports)
        py_ports = [
            PortInitializer(
                pt.name, pt.shape, getattr(ProcModel, pt.name).d_type,
                pt.__class__.__name__, 32
            )
            for pt in ports
        ]

        # But do not assign all of them to builder
        b.set_variables(v[:-1])
        b.set_py_ports(py_ports[:-1])

        # Before a builder it deployed to a remote node, the compiler will
        # check if Var- and PortInitializers have been set for all LavaPyTypes.
        # Thus his will fails.
        with self.assertRaises(AssertionError):
            b.check_all_vars_and_ports_set()

        # But when we set all of them...
        b.set_variables([v[-1]])
        b.set_py_ports([py_ports[-1]])

        # ... the check will pass
        b.check_all_vars_and_ports_set()

    def test_check_lava_py_types(self):
        """Checks identification of illegal LavaPyType settings.

        All ProcModels tested here implement ProcForLavaPyType which has one
        InPort called 'port'
        """

        # Create universal PortInitializer reflecting the 'port' in
        # ProcForLavaPyType
        pi = PortInitializer("port", (1,), np.intc, "InPort", 32)

        # Create PortInitializer for correct LavaPyType(PyInPort.VEC_DENSE, int)
        b = PyProcessBuilder(ProcModelForLavaPyType0, 0)
        b.set_py_ports([pi])

        # This one is legal
        b.check_lava_py_types()

        # Create PortInitializer for wrong LavaPyType(123, int)
        b = PyProcessBuilder(ProcModelForLavaPyType1, 1)
        b.set_py_ports([pi])

        # This one fails because '123' is not a type
        with self.assertRaises(AssertionError):
            b.check_lava_py_types()

        # Create PortInitializer for wrong LavaPyType(PyInPort, int)
        b = PyProcessBuilder(ProcModelForLavaPyType2, 2)
        b.set_py_ports([pi])

        # This one fails because 'PyInPort' is not a strict subtype of
        # 'PyInPort'
        with self.assertRaises(AssertionError):
            b.check_lava_py_types()

        # Create PortInitializer for wrong LavaPyType(PyOutPort, int)
        b = PyProcessBuilder(ProcModelForLavaPyType3, 3)
        b.set_py_ports([pi])

        # This one fails because 'PyOutPort' is not a strict sub-type at all
        # of PyInPort
        with self.assertRaises(AssertionError):
            b.check_lava_py_types()

    def test_build(self):
        """Checks building of ProcessModel by builder."""

        # Lets create a ProcBuilder and Proc with Var- and PortInitializers
        # and fake CspPorts so ProcModel can be built
        b = PyProcessBuilder(ProcModel, 0)

        proc = Proc()
        v = [VarInitializer(v.name, v.shape, v.init, v.id) for v in proc.vars]

        ports = list(proc.in_ports) + list(proc.out_ports)
        py_ports = [
            PortInitializer(
                pt.name, pt.shape, getattr(ProcModel, pt.name).d_type,
                pt.__class__.__name__, 32
            )
            for pt in ports
        ]

        csp_ports = []
        for py_port in py_ports:
            csp_ports.append(FakeCspPort(py_port.name))

        # Set all Var-, PortInitializers and fake CspPorts
        b.set_variables(v)
        b.set_py_ports(py_ports)
        b.set_csp_ports(csp_ports)

        # Before we build, we should make sure all vars and ports are set and
        # that there's a CSP port for every PyPort
        b.check_all_vars_and_ports_set()

        # Building the ProcModel will initialize Vars and Ports on ProcModel
        # instance
        pm = b.build()

        # Thus the ProcModel instance should have PyPort attributes as
        # defined by by the Process and ProcessModel class
        import lava.magma.core.model.py.ports as pts

        self.assertTrue(hasattr(pm, "in_port"))
        self.assertIsInstance(pm.in_port, pts.PyInPortVectorDense)
        self.assertTrue(hasattr(pm, "out_port"))
        self.assertIsInstance(pm.out_port, pts.PyOutPortVectorDense)
        # And these ports should have the specified shape
        self.assertEqual(pm.in_port._shape, (2, 1))
        self.assertEqual(pm.out_port._shape, (3, 2))

        # Similarly, Var attributes should exist with all initial values
        # being broadcast to required shape if necessary
        self.assertEqual(pm.v1_scalar, 0)
        self.assertEqual(pm.v2_scalar_init, 2)
        self.assertTrue(
            np.array_equal(
                pm.v3_tensor_broadcast, np.ones((2, 3), dtype=np.int32) * 10
            )
        )
        self.assertTrue(
            np.array_equal(
                pm.v4_tensor, np.array(
                    [[1, 2], [3, 4], [5, 6]], dtype=np.int32)
            )
        )

        # In addition, private attribute for variables precisions got created
        # following the naming convention "_<var_name>_p":
        self.assertEqual(pm._v1_scalar_p, 27)
        self.assertEqual(pm._v2_scalar_init_p, 27)
        self.assertEqual(pm._v3_tensor_broadcast_p, 6)
        self.assertEqual(pm._v4_tensor_p, 6)

        # Just to make sure the generated Vars are really usable, we can call
        # the run(..) method:
        self.assertEqual(pm.run(), 1)

    def test_build_with_dangling_ports(self):
        """Checks that not all ports must be connected, i.e. ports can be
        left dangling."""

        # First create a process with no OutPorts
        proc_with_no_out_ports = Proc()
        v = [VarInitializer(v.name, v.shape, v.init, v.id)
             for v in proc_with_no_out_ports.vars]

        ports = \
            list(proc_with_no_out_ports.in_ports) + \
            list(proc_with_no_out_ports.out_ports)
        py_ports = [
            PortInitializer(
                pt.name, pt.shape, getattr(ProcModel, pt.name).d_type,
                pt.__class__.__name__, 32
            )
            for pt in ports
        ]

        csp_ports = []
        for py_port in list(proc_with_no_out_ports.in_ports):
            csp_ports.append(FakeCspPort(py_port.name))

        b_with_no_out_ports = PyProcessBuilder(ProcModel, 0)
        b_with_no_out_ports.set_variables(v)
        b_with_no_out_ports.set_py_ports(py_ports)
        b_with_no_out_ports.set_csp_ports(csp_ports)

        # Next create a process with no InPorts
        proc_with_no_in_ports = Proc()
        v = [VarInitializer(v.name, v.shape, v.init, v.id)
             for v in proc_with_no_in_ports.vars]

        ports = \
            list(proc_with_no_in_ports.in_ports) + \
            list(proc_with_no_in_ports.out_ports)
        py_ports = [
            PortInitializer(
                pt.name, pt.shape, getattr(ProcModel, pt.name).d_type,
                pt.__class__.__name__, 32
            )
            for pt in ports
        ]

        csp_ports = []
        for py_port in list(proc_with_no_in_ports.out_ports):
            csp_ports.append(FakeCspPort(py_port.name))

        b_with_no_in_ports = PyProcessBuilder(ProcModel, 0)
        b_with_no_in_ports.set_variables(v)
        b_with_no_in_ports.set_py_ports(py_ports)
        b_with_no_in_ports.set_csp_ports(csp_ports)

        # Validate builders
        b_with_no_out_ports.check_all_vars_and_ports_set()
        b_with_no_in_ports.check_all_vars_and_ports_set()

        # Then build them
        pm_with_no_out_ports = b_with_no_out_ports.build()
        pm_with_no_in_ports = b_with_no_in_ports.build()

        # Validate that the Process with no OutPorts indeed has no output
        # CspPort
        self.assertIsInstance(
            pm_with_no_out_ports.in_port.csp_ports[0], FakeCspPort)
        self.assertEqual(pm_with_no_out_ports.out_port.csp_ports, [])

        # Validate that the Process with no InPorts indeed has no input
        # CspPort
        self.assertEqual(pm_with_no_in_ports.in_port.csp_ports, [])
        self.assertIsInstance(
            pm_with_no_in_ports.out_port.csp_ports[0], FakeCspPort)

    def test_set_ref_var_ports(self):
        """Check RefPorts and VarPorts can be set."""

        # Create a new ProcBuilder
        b = PyProcessBuilder(PyProcModelRefVar, 0)

        # Create Process for which we want to build PyProcModel
        proc = ProcRefVar()

        # Normally, the Compiler would create PortInitializers from all
        # ref ports holding only its name and shape
        ports = list(proc.ref_ports)
        ref_ports = [PortInitializer(
            pt.name,
            pt.shape,
            getattr(PyProcModelRefVar, pt.name).d_type,
            pt.__class__.__name__, 32)
            for pt in ports]
        # Similarly, the Compiler would create VarPortInitializers from all
        # var ports holding only its name, shape and var_name
        ports = list(proc.var_ports)
        var_ports = [VarPortInitializer(
            pt.name,
            pt.shape,
            pt.var.name,
            getattr(PyProcModelRefVar, pt.name).d_type,
            pt.__class__.__name__, 32, PyRefPort.VEC_DENSE)
            for pt in ports]
        # The Runtime, would normally create CspPorts that implement the actual
        # message passing via channels between RefPorts and VarPorts. Here we
        # just create some fake CspPorts for each Ref- and VarPort.
        # 2 CspChannels per Ref-/VarPort.
        csp_ports = []
        for port in list(ref_ports):
            csp_ports.append(FakeCspPort(port.name))
            csp_ports.append(FakeCspPort(port.name))
        for port in list(var_ports):
            csp_ports.append(FakeCspPort(port.name))
            csp_ports.append(FakeCspPort(port.name))

        # During compilation, the Compiler creates and then sets
        # PortInitializers and VarPortInitializers
        b.set_ref_ports(ref_ports)
        b.set_var_ports(var_ports)
        # The Runtime sets CspPorts
        b.set_csp_ports(csp_ports)

        # All the objects are converted into dictionaries to retrieve them by
        # name
        self.assertEqual(list(b.py_ports.values()), [])
        self.assertEqual(list(b.ref_ports.values()), ref_ports)
        self.assertEqual(list(b.var_ports.values()), var_ports)
        self.assertEqual(list(v for vv in b.csp_ports.values()
                              for v in vv), csp_ports)
        self.assertEqual(b.ref_ports["ref"], ref_ports[0])
        self.assertEqual(b.csp_ports["ref"], [csp_ports[0], csp_ports[1]])
        self.assertEqual(b.var_ports["var_port"], var_ports[0])
        self.assertEqual(b.csp_ports["var_port"], [csp_ports[2], csp_ports[3]])


if __name__ == "__main__":
    unittest.main()
