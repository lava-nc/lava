# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import logging
import typing as ty
import unittest
from unittest.mock import Mock

from lava.magma.compiler.builders.channel_builder import ChannelBuilderMp
from lava.magma.compiler.channel_map import ChannelMap, Payload, PortPair
from lava.magma.compiler.channels.interfaces import ChannelType
from lava.magma.compiler.compiler_graphs import ProcGroupDiGraphs
from lava.magma.compiler.subcompilers.channel_builders_factory import \
    ChannelBuildersFactory
from lava.magma.compiler.utils import LoihiPortInitializer, PortInitializer
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.interfaces import AbstractPortImplementation
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.ports import (PyInPort, PyOutPort, PyRefPort,
                                            PyVarPort)
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import (AbstractPort, InPort, OutPort,
                                                 RefPort, VarPort)
from lava.magma.core.process.ports.reduce_ops import ReduceSum
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.sync.protocol import AbstractSyncProtocol


# A minimal process (A) with an InPort, OutPort and RefPort
class ProcA(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use ReduceOp to allow for multiple input connections
        self.inp = InPort(shape=(1,), reduce_op=ReduceSum)
        self.out = OutPort(shape=(1,))
        self.ref = RefPort(shape=(10,))


# Another minimal process (B) with a Var and an InPort, OutPort and VarPort
class ProcB(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use ReduceOp to allow for multiple input connections
        self.inp = InPort(shape=(1,), reduce_op=ReduceSum)
        self.out = OutPort(shape=(1,))
        self.var = Var((10,), init=10)
        self.var_port = VarPort(self.var)


# Another minimal process (C) with an InPort and OutPort
class ProcC(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use ReduceOp to allow for multiple input connections
        self.inp = InPort(shape=(1,), reduce_op=ReduceSum)
        self.out = OutPort(shape=(1,))


class MockRuntimeService:
    __name__ = "MockRuntimeService"


# Define minimal Protocol to be implemented
class ProtocolA(AbstractSyncProtocol):
    @property
    def runtime_service(self):
        return {CPU: MockRuntimeService()}


# Define minimal Protocol to be implemented
class ProtocolB(AbstractSyncProtocol):
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


# A minimal PyProcModel implementing ProcB
@implements(ProcB, protocol=ProtocolB)
@requires(CPU)
class PyProcModelB(AbstractPyProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
    var: int = LavaPyType(int, int)
    var_port: PyVarPort = LavaPyType(PyVarPort.VEC_DENSE, int)

    def run(self):
        pass


# A minimal PyProcModel implementing ProcC
@implements(proc=ProcC)
@requires(CPU)
class PyProcModelC(AbstractPyProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run(self):
        pass


# A minimal SubProcModel implementing ProcA using ProcB as sub processes
# which we consider a LeafProcess that has no further sub process decomposition
@implements(ProcA)
class SubProcModelA(AbstractSubProcessModel):
    def __init__(self, proc):
        self.proc1 = ProcB()
        self.proc2 = ProcC()
        self.proc1.out.connect(self.proc2.inp)
        proc.inp.connect(self.proc1.inp)
        self.proc2.out.connect(proc.out)

    def run(self):
        pass


# A minimal RunConfig that will select SubProcModel if there is one
class RunCfg(RunConfig):
    def __init__(
        self,
        loglevel: int = logging.WARNING,
        custom_sync_domains=None,
        select_sub_proc_model=False,
        select_lmt: bool = False,
    ):
        super().__init__(
            custom_sync_domains=custom_sync_domains, loglevel=loglevel
        )
        self.select_sub_proc_model = select_sub_proc_model
        self.select_lmt = select_lmt

    def select(self, proc, proc_models):
        py_proc_model = None
        sub_proc_model = None

        # Find PyProcModel or SubProcModel
        for pm in proc_models:
            if issubclass(pm, AbstractSubProcessModel):
                sub_proc_model = pm
            if issubclass(pm, AbstractPyProcessModel):
                py_proc_model = pm
        # Make selection
        if self.select_sub_proc_model and sub_proc_model:
            return sub_proc_model
        if py_proc_model and not self.select_lmt:
            return py_proc_model
        else:
            raise AssertionError("No legal ProcessModel found.")


class PortInitializerFactory:
    """Factory for port initializers given a port type."""

    OrderedPortDict = ty.OrderedDict[
        ty.Type[AbstractPortImplementation], ty.Type[PortInitializer]
    ]
    port_to_initializer_type = OrderedPortDict(
        {
            AbstractPyProcessModel: PortInitializer,
        }
    )

    @staticmethod
    def initializer_from_port(port: AbstractPort, *args) -> PortInitializer:
        """Create a PortInitializer for given the port.

        Parameters
        ----------
        port: The port to be associated with the initializer.

        Returns
        -------
        an instance of PortInitializer initialized with args.
        """
        port_class = ChannelBuildersFactory._get_port_process_model_class(port)
        cls = None
        for cls in reversed(port_class.__mro__):
            if cls in PortInitializerFactory.port_to_initializer_type.keys():
                break
        if cls is None:
            str_msg = PortInitializerFactory.port_to_initializer_type.keys()
            raise KeyError(f"The port class is not in {str_msg}.")
        port_init = PortInitializerFactory.port_to_initializer_type[cls]
        port_init = LoihiPortInitializer(
            port.name,
            port.shape,
            ChannelBuildersFactory.get_port_dtype(port),
            port.__class__.__name__,
            64,
            port.get_incoming_transform_funcs(),
        )
        return port_init

    @staticmethod
    def set_initializers_on_channel_map(channel_map: ChannelMap):
        for port in PortInitializerFactory.get_unique_ports_from_channel_map(
            channel_map
        ):
            port_init = PortInitializerFactory.initializer_from_port(port)
            channel_map.set_port_initializer(port, port_init)

    def get_unique_ports_from_channel_map(channel_map: ChannelMap):
        ports = [
            port for pair in channel_map.keys() for port in [pair.src, pair.dst]
        ]
        return set(ports)


class TestChannelBuildersFactory(unittest.TestCase):
    def setUp(self) -> None:
        self.proc_a = ProcA()
        self.proc_b = ProcB()
        self.proc_a._model_class = PyProcModelA
        self.proc_b._model_class = PyProcModelB
        self.proc_a.out.connect(self.proc_b.inp)
        self.proc_b.out.connect(self.proc_a.inp)
        port_pair1 = PortPair(src=self.proc_a.out, dst=self.proc_b.inp)
        port_pair2 = PortPair(src=self.proc_b.out, dst=self.proc_a.inp)
        self.channel_map_mock = Mock(spec_set=ChannelMap)
        self.channel_map_mock.keys.return_value = [port_pair1, port_pair2]
        self.channel_map = ChannelMap()
        self.channel_map[port_pair1] = Payload(multiplicity=1)
        self.channel_map[port_pair2] = Payload(multiplicity=1)
        PortInitializerFactory.set_initializers_on_channel_map(self.channel_map)
        self.channel_builder_factory = ChannelBuildersFactory()
        self.cfg = {"pypy_channel_size": 64}

    def test_obj_creation(self):
        self.assertIsInstance(
            self.channel_builder_factory, ChannelBuildersFactory
        )

    def test_compile_method_accepts_and_uses_ChannelMap(self):
        self.channel_builder_factory.from_channel_map(
            self.channel_map_mock, self.cfg
        )
        self.channel_map_mock.keys.assert_called_once()

    def test_compile_does_not_have_side_effects_on_channel_map(self):
        chm_pre = self.channel_map.copy()
        self.channel_builder_factory.from_channel_map(
            self.channel_map, self.cfg
        )
        self.assertEqual(chm_pre, self.channel_map)

    def test_get_builders_return_a_list_of_channel_builders(self):
        output = self.channel_builder_factory.from_channel_map(
            self.channel_map, self.cfg
        )
        self.assertIsInstance(output, list)
        self.assertIsInstance(output[0], ChannelBuilderMp)

    def test_every_channel_builder_has_not_none_port_initializers(self):
        builders = self.channel_builder_factory.from_channel_map(
            self.channel_map, self.cfg
        )
        self.assertGreater(len(builders), 0)
        for builder in builders:
            self.assertIsNotNone(builder.src_port_initializer)
            self.assertIsNotNone(builder.dst_port_initializer)

    def test_unique_channel_builder_exist_per_channel_with_a_pyport(self):
        builders = self.channel_builder_factory.from_channel_map(
            self.channel_map, self.cfg
        )
        self.assertEqual(len(builders), 2)
        self.assertNotEqual(builders[0], builders[1])
        for builder in builders:
            condition_1_1 = builder.src_process == self.proc_a
            condition_1_2 = builder.dst_process == self.proc_b
            condition_2_1 = builder.src_process == self.proc_b
            condition_2_2 = builder.dst_process == self.proc_a
            condition1 = condition_1_1 and condition_1_2
            condition2 = condition_2_1 and condition_2_2
            self.assertTrue(condition1 or condition2)

    def test_no_channel_was_created_for_keys_not_involving_py_ports(self):
        builders = self.channel_builder_factory.from_channel_map(
            self.channel_map, self.cfg
        )
        channel_types = {ChannelType.PyPy}
        outcome = {builder.channel_type for builder in builders}
        self.assertEqual(outcome, channel_types)

    def test_input_to_channel_builder_is_not_none(self):
        builders = self.channel_builder_factory.from_channel_map(
            self.channel_map, self.cfg
        )
        for builder in builders:
            self.assertNotEqual(builder.src_process, None)
            self.assertNotEqual(builder.dst_process, None)
            self.assertNotEqual(builder.dst_port_initializer, None)
            self.assertNotEqual(builder.dst_port_initializer, None)
            self.assertNotEqual(builder.channel_type, None)


class TestChannelBuildersMp(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {"pypy_channel_size": 64}
        self.factory = ChannelBuildersFactory()

    def test_create_channel_builders(self):
        """Checks creation of ChannelBuilders.

        This test does currently only support PyPyChannels.
        """

        # Create a set of processes...
        p1, p2, p3, p4, p5 = ProcA(), ProcB(), ProcC(), ProcA(), ProcB()

        # ...and connect them in linear and circular manner so we have
        # dangling connections, forks, and joins
        # p1 -> p2 -> p3 -> p4 -> (p2, p5)
        p1.out.connect(p2.inp)
        p2.out.connect(p3.inp)
        p3.out.connect(p4.inp)
        p4.out.connect([p2.inp, p5.inp])

        # Create a manual proc_group
        proc_group = []
        pm_mapping = {
            p1: PyProcModelA,
            p2: PyProcModelB,
            p3: PyProcModelC,
            p4: PyProcModelA,
            p5: PyProcModelB,
        }
        for p, pm in pm_mapping.items():
            p._model_class = pm
            proc_group.append(p)

        # Create channel builders
        channel_map = ChannelMap.from_proc_groups([proc_group])
        PortInitializerFactory.set_initializers_on_channel_map(channel_map)
        channel_builders = self.factory.from_channel_map(channel_map, self.cfg)

        # This should result in 5 channel builders (one for each arrow above)
        self.assertEqual(len(channel_builders), 5)
        for cb in channel_builders:
            self.assertIsInstance(cb, ChannelBuilderMp)

        # Each channel builder should connect its source and destination
        # process and port
        # Let's check the first one in detail
        self.assertEqual(channel_builders[0].channel_type, ChannelType.PyPy)
        self.assertEqual(channel_builders[0].src_process, p1)
        self.assertEqual(channel_builders[0].src_port_initializer.name, "out")
        self.assertEqual(channel_builders[0].src_port_initializer.shape, (1,))
        self.assertEqual(channel_builders[0].src_port_initializer.d_type, int)
        self.assertEqual(channel_builders[0].src_port_initializer.size, 64)
        self.assertEqual(channel_builders[0].dst_process, p2)
        # ... and for others only the source/destination processes
        self.assertEqual(channel_builders[1].src_process, p2)
        self.assertEqual(channel_builders[1].dst_process, p3)
        self.assertEqual(channel_builders[2].src_process, p3)
        self.assertEqual(channel_builders[2].dst_process, p4)
        self.assertEqual(channel_builders[3].src_process, p4)
        self.assertEqual(channel_builders[3].dst_process, p2)
        self.assertEqual(channel_builders[4].src_process, p4)
        self.assertEqual(channel_builders[4].dst_process, p5)

    def test_create_channel_builders_hierarchical_process(self):
        """Check creation of channel builders when a SubProcModel is used
        with connections potentially terminating in dangling ports of
        hierarchical processes."""

        # Create a minimal process
        p = ProcA()

        # Manually map processes to sub processes to internal sub structure
        # gets build
        proc_group_digraph = ProcGroupDiGraphs(
            p, RunCfg(select_sub_proc_model=True)
        )
        proc_groups = proc_group_digraph.get_proc_groups()

        p._model_class = SubProcModelA

        channel_map = ChannelMap.from_proc_groups(proc_groups)
        PortInitializerFactory.set_initializers_on_channel_map(channel_map)
        channel_builders = self.factory.from_channel_map(channel_map, self.cfg)

        # There should only be one ChannelBuilder from the internal proc1 to
        # proc2
        self.assertEqual(len(channel_builders), 1)
        from lava.magma.compiler.builders.channel_builder import \
            ChannelBuilderMp

        self.assertIsInstance(channel_builders[0], ChannelBuilderMp)
        self.assertEqual(channel_builders[0].src_process, p.procs.proc1)
        self.assertEqual(channel_builders[0].dst_process, p.procs.proc2)

    def test_create_channel_builders_ref_ports(self):
        """Checks creation of channel builders when a process is connected
        using a RefPort to another process VarPort."""

        # Create a process with a RefPort (source)
        src = ProcA()

        # Create a process with a var (destination)
        dst = ProcB()

        # Connect them using RefPort and VarPort
        proc_group = []
        src.ref.connect(dst.var_port)

        # Create a manual proc_map
        pm_mapping = {src: PyProcModelA, dst: PyProcModelB}
        for p, pm in pm_mapping.items():
            p._model_class = pm
            proc_group.append(p)

        # Create channel builders
        channel_map = ChannelMap.from_proc_groups([proc_group])
        PortInitializerFactory.set_initializers_on_channel_map(channel_map)
        channel_builders = self.factory.from_channel_map(channel_map, self.cfg)
        for p, pm in pm_mapping.items():
            p._model_class = pm
            proc_group.append(p)

        # This should result in 2 channel builder
        from lava.magma.compiler.builders.channel_builder import \
            ChannelBuilderMp

        self.assertEqual(len(channel_builders), 2)
        self.assertIsInstance(channel_builders[0], ChannelBuilderMp)
        self.assertEqual(channel_builders[0].src_process, src)
        self.assertEqual(channel_builders[0].dst_process, dst)

    def test_create_channel_builders_ref_ports_implicit(self):
        """Checks creation of channel builders when a process is connected
        using a RefPort to another process Var (implicit VarPort)."""

        # Create a process with a RefPort (source)
        src = ProcA()

        # Create a process with a var (destination)
        dst = ProcB()

        # Connect them using RefPort and Var (creates implicitly a VarPort)
        src.ref.connect_var(dst.var)

        # Create a manual proc_map
        proc_group = []
        pm_mapping = {src: PyProcModelA, dst: PyProcModelB}

        for p, pm in pm_mapping.items():
            p._model_class = pm
            proc_group.append(p)

        # Create channel builders
        channel_map = ChannelMap.from_proc_groups([proc_group])
        PortInitializerFactory.set_initializers_on_channel_map(channel_map)
        channel_builders = self.factory.from_channel_map(channel_map, self.cfg)

        # This should result in 2 channel builder
        from lava.magma.compiler.builders.channel_builder import \
            ChannelBuilderMp

        self.assertEqual(len(channel_builders), 2)
        self.assertIsInstance(channel_builders[0], ChannelBuilderMp)
        self.assertEqual(channel_builders[0].src_process, src)
        self.assertEqual(channel_builders[0].dst_process, dst)


if __name__ == "__main__":
    unittest.main()
