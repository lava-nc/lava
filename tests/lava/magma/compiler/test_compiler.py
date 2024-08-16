# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import logging
import typing as ty
import unittest
from unittest.mock import Mock, PropertyMock, patch, seal

import lava.magma.compiler.exceptions as ex
from lava.magma.compiler.builders.channel_builder import ChannelBuilderMp
from lava.magma.compiler.builders.interfaces import AbstractProcessBuilder

from lava.magma.compiler.builders.py_builder import PyProcessBuilder
from lava.magma.compiler.channel_map import ChannelMap
from lava.magma.compiler.compiler import Compiler
from lava.magma.compiler.compiler_graphs import ProcGroup
from lava.magma.compiler.subcompilers.py.pyproc_compiler import PyProcCompiler
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.ports import (PyInPort, PyOutPort, PyRefPort,
                                            PyVarPort)
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import (InPort, OutPort, RefPort,
                                                 VarPort)
from lava.magma.core.process.ports.reduce_ops import ReduceSum
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol


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
        self.some_var = Var((10,), init=10)
        self.var_port = VarPort(self.some_var)


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
    some_var: int = LavaPyType(int, int)
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
    ):
        super().__init__(
            custom_sync_domains=custom_sync_domains, loglevel=loglevel
        )
        self.select_sub_proc_model = select_sub_proc_model

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
        elif py_proc_model:
            return py_proc_model
        else:
            raise AssertionError("No legal ProcessModel found.")


def create_mock_proc_groups() -> ty.List[ProcGroup]:
    """Creates a list of mock ProcGroups to set up multiple test cases."""
    py_pg1_p1 = Mock(spec_set=AbstractProcess())
    py_pg2_p1 = Mock(spec_set=AbstractProcess())
    py_pg2_p2 = Mock(spec_set=AbstractProcess())

    py_pg1_p1.configure_mock(name='py_pg1_p1')
    py_pg2_p1.configure_mock(name='py_pg2_p1')
    py_pg2_p2.configure_mock(name='py_pg2_p2')

    proc_list = [py_pg1_p1, py_pg2_p1, py_pg2_p2]

    proc_model_types = [
        AbstractPyProcessModel,
        AbstractPyProcessModel,
        AbstractPyProcessModel
    ]

    # Add mock PyProcModel instances to each of the Processes.
    for p, pm_type in zip(proc_list, proc_model_types):
        type(p).model_class = PropertyMock(return_value=pm_type)
        seal(p)

    proc_groups = [
        [py_pg1_p1],  # single Process
        [py_pg2_p1, py_pg2_p2],  # homogeneousProcGroup
    ]

    return proc_groups


def create_builders(
    proc_groups: ty.List[ty.List[AbstractProcess]],
) -> ty.List[ty.Dict]:
    """Creates the sequence of return values for the get_builders() method of
    the three SubCompilers, given the ProcGroups created in the
    create_proc_groups function."""
    py_builders = [
        {proc_groups[0][0]: Mock(spec_set=PyProcessBuilder)},
        {proc_groups[1][0]: Mock(spec_set=PyProcessBuilder)}
    ]

    return py_builders


def create_patches(
    py_builders: ty.List[ty.Dict]
) -> patch:
    """Creates patches (unittest.mock.patch) for the three SubCompilers such
    that their get_builders() methods return the correct sequence of Builders
    and the compile() method returns the given ChannelMap unchanged.
    ."""

    def compile_return(channel_map: ChannelMap,
                       partitioning=None) -> ChannelMap:
        return channel_map

    py_patch = patch(
        "lava.magma.compiler.compiler.PyProcCompiler",
        spec_set=PyProcCompiler,
        get_builders=Mock(side_effect=GetBuilders(py_builders)),
        compile=Mock(side_effect=compile_return),
    )

    return py_patch


class GetBuilders:
    """Returns Builders in the same order in which the real
    implementation would return them."""

    def __init__(
        self,
        builders: ty.List[ty.Dict[AbstractProcess, AbstractProcessBuilder]]
    ) -> None:
        self._count = 0
        self._builders = builders

    def __call__(self, channel_map: ChannelMap):
        return_value = self._builders[self._count]
        self._count += 1
        return return_value, channel_map


class TestCompiler(unittest.TestCase):
    def setUp(self) -> None:
        """Create an instance of a Compiler."""
        self.compiler = Compiler()

    def test_init_creates_compiler_instance(self) -> None:
        """Tests initialization of the Compiler."""
        self.assertIsInstance(self.compiler, Compiler)

    def test_init_arguments(self) -> None:
        """Tests the arguments of the __init__ method."""
        compile_config = {"foo": "bar"}
        loglevel = logging.DEBUG
        compiler = Compiler(compile_config=compile_config, loglevel=loglevel)
        self.assertEqual(compiler._compile_config, compile_config)
        self.assertEqual(compiler.log.level, loglevel)

    def test_map_subcomp_from_homogeneous_proc_group(self) -> None:
        """Tests _map_compiler_types_to_procs() with a ProcGroup
        that consists of Processes that all have the same
        ProcessModel type and thus require the same type of
        subcompiler to be compiled."""
        # Create a ProcGroup with three stub Processes.
        proc_group = [
            Mock(spec_set=AbstractProcess),
            Mock(spec_set=AbstractProcess),
            Mock(spec_set=AbstractProcess),
        ]

        # Add mock PyProcModel instances to each of the Processes.
        for p in proc_group:
            type(p).model_class = PropertyMock(
                return_value=AbstractPyProcessModel
            )

        mapping = self.compiler._map_subcompiler_type_to_procs(proc_group)

        # There should only be a single entry in the mapping...
        self.assertEqual(len(mapping), 1)
        # ...which is for a subcompiler of type PyProcCompiler, ...
        self.assertTrue(mapping[PyProcCompiler])
        # ...which has a list...
        self.assertIsInstance(mapping[PyProcCompiler], list)
        # ...with three entries
        self.assertEqual(len(mapping[PyProcCompiler]), 3)
        # ...and the list is the same as proc_group.
        self.assertListEqual(mapping[PyProcCompiler], proc_group)

    def test_map_subcomp_from_inhomogeneous_proc_group(self) -> None:
        """Tests _map_compiler_types_to_procs() with a ProcGroup
        that consists of Processes that have different ProcessModel types
        (covering all available ones) and thus require different subcompiler
        types to be compiled."""
        # Create a ProcGroup with three mock Processes.
        proc_group = [
            Mock(spec_set=AbstractProcess)
        ]

        proc_model_types = [
            AbstractPyProcessModel
        ]

        # Add mock PyProcModel instances to each of the Processes.
        for p, pm_type in zip(proc_group, proc_model_types):
            type(p).model_class = PropertyMock(return_value=pm_type)

        mapping = self.compiler._map_subcompiler_type_to_procs(proc_group)

        # There should be three entries in the mapping...
        self.assertEqual(len(mapping), 1)
        # ...one for each type of subcompiler...
        self.assertTrue(mapping[PyProcCompiler])
        # ...and each entry should contain a list with a single Process,
        # namely the one that has the ProcessModel type associated with the
        # Subcompiler type.
        self.assertListEqual(mapping[PyProcCompiler], [proc_group[0]])

    def test_map_subcomp_unsupported_proc_model_raises_exception(self) -> None:
        """Tests _map_compiler_types_to_procs() with a ProcGroup
        that contains a Process with a ProcessModel, for which there
        is no SubCompiler available. -> NotImplementedError"""
        # Create a ProcGroup with two mock Processes.
        proc_group = [
            Mock(spec_set=AbstractProcess),
            Mock(spec_set=AbstractProcess),
        ]

        class UnsupportedProcessModel:
            pass

        proc_model_types = [AbstractPyProcessModel, UnsupportedProcessModel]

        # Add mock PyProcModel instances to each of the Processes.
        for p, pm_type in zip(proc_group, proc_model_types):
            type(p).model_class = PropertyMock(return_value=pm_type)

        with self.assertRaises(NotImplementedError):
            self.compiler._map_subcompiler_type_to_procs(proc_group)

    def test_create_subcompilers_initializes_all_subcompilers(self) -> None:
        """Tests that the method _create_subcompilers() calls the init method
        for all SubCompiler classes in the given dictionary."""
        # Create mocks for the SubCompiler classes (not their instances).
        PyProcCompilerMock = Mock(spec_set=type(PyProcCompiler))

        # For each SubCompiler, create a list of Processes.
        py_procs = [Mock(spec_set=AbstractProcess)]

        # Create a mapping as would be returned by the method
        # _map_compiler_type_to_procs() from SubCompilerClass to its list of
        # Processes.
        mapping = {
            PyProcCompilerMock: py_procs,
        }

        # Create the subcompiler instances.
        subcompilers = self.compiler._create_subcompilers(mapping)

        # Check that the __init__ method was called on every mock SubCompiler
        # with its corresponding list of Processes and the Compiler's
        # compile_config.
        cfg = self.compiler._compile_config
        PyProcCompilerMock.assert_called_with(py_procs, cfg)

        # Also check that the returned list of SubCompilers has the correct
        # length.
        self.assertEqual(len(subcompilers), 1)

    def test_compile_proc_group_single_loop(self) -> None:
        """Test whether the correct methods are called on all objects when
        the compilation loop makes a single iteration."""
        channel_map = ChannelMap()

        # Create mock SubCompiler instances with a mock compile() method that
        # returns an unchanged ChannelMap.
        py_proc_compiler = Mock(
            spec_set=PyProcCompiler, compile=Mock(return_value=channel_map)
        )

        seal(py_proc_compiler)

        subcompilers = [py_proc_compiler]

        # Call the method to be tested.
        self.compiler._compile_proc_group(subcompilers, channel_map, None)

        # Check that it called compile() on every SubCompiler instance
        # exactly once. After that, the while loop should exit because the
        # ChannelMap instance has not changed.
        for sc in subcompilers:
            sc.compile.assert_called_once_with({}, None)

    def test_compile_proc_group_multiple_loops(self) -> None:
        """Test whether the correct methods are called on all objects when
        the compilation loop makes multiple iterations."""
        channel_map = ChannelMap()

        # Return values of compile() methods.
        channel_map1 = ChannelMap(
            {(Mock(spec_set=InPort), Mock(spec_set=OutPort)): 1}
        )
        channel_map2 = channel_map1.copy()
        channel_map2.update(
            {(Mock(spec_set=InPort), Mock(spec_set=OutPort)): 1}
        )
        channel_map3 = channel_map2.copy()

        # Create mock SubCompiler instances with a mock compile() method that
        # returns the given sequence of ChannelMaps.
        return_values = [channel_map1, channel_map2, channel_map3]
        py_proc_compiler = Mock(
            spec_set=PyProcCompiler, compile=Mock(side_effect=return_values)
        )
        seal(py_proc_compiler)
        subcompilers = [py_proc_compiler]

        # Call the method to be tested.
        self.compiler._compile_proc_group(subcompilers, channel_map,
                                          None)

        # Check that it called compile() on every SubCompiler instance
        # exactly once. After that, the while loop should exit because the
        # ChannelMap instance has not changed.
        for sc in subcompilers:
            sc.compile.assert_called_with({**channel_map1, **channel_map2},
                                          None)
            self.assertEqual(sc.compile.call_count, 3)

    def test_extract_proc_builders(self) -> None:
        """Tests whether the Executable is populated with the correct Builder
        instances."""
        # Create some mock Processes.
        proc1 = Mock(spec_set=AbstractProcess)
        proc2 = Mock(spec_set=AbstractProcess)

        # Create some Builders.
        py_builder1 = PyProcessBuilder(AbstractPyProcessModel, 0)
        py_builder2 = PyProcessBuilder(AbstractPyProcessModel, 0)

        # Create mock SubCompilers, covering each type (Py, C, Nc).
        # Each SubCompiler has a get_builders() method that returns a mapping
        # from a Process to a Builder.
        # The two PyProcCompilers mimic Processes from different ProcGroups.
        py_proc_compiler1 = Mock(
            spec_set=PyProcCompiler,
            get_builders=Mock(
                return_value=({proc1: py_builder1}, ChannelMap())
            ),
        )
        seal(py_proc_compiler1)
        py_proc_compiler2 = Mock(
            spec_set=PyProcCompiler,
            get_builders=Mock(
                return_value=({proc2: py_builder2}, ChannelMap())
            ),
        )
        seal(py_proc_compiler2)

        subcompilers = [
            py_proc_compiler1,
            py_proc_compiler2
        ]

        # Call the method to be tested.
        channel_map = ChannelMap()
        proc_builders, channel_map = self.compiler._extract_proc_builders(
            subcompilers, channel_map
        )

        # Inside the method, the builders of each compiler must have been
        # extracted.
        for sc in subcompilers:
            sc.get_builders.assert_called_once()

        py_builders = {}
        for proc, builder in proc_builders.items():
            entry = {proc: builder}
            if isinstance(builder, PyProcessBuilder):
                py_builders.update(entry)

        # There should be two PyProcBuilders...
        self.assertEqual(len(py_builders), 2)
        self.assertEqual(py_builders[proc1], py_builder1)
        self.assertEqual(py_builders[proc2], py_builder2)

    def test_compile_proc_groups(self) -> None:
        """Tests the method _compile_proc_groups() with a list of
        ProcGroups that covers a ProcGroup with a single Process,
        a homogeneous ProcGroup with Processes that share the same
        ProcModel type, and an inhomogeneous ProcGroup with Processes that
        have different ProcModel types."""

        # Create a list of mock ProcGroups.
        proc_groups = create_mock_proc_groups()
        # Create the list of Builders that the
        py_builders = create_builders(proc_groups)
        # Create patches for the PyProcCompiler
        py_patch = create_patches(
            py_builders
        )

        # Patch the SubCompilers and their get_builders() method.
        channel_map = ChannelMap()
        with py_patch:
            # Call the method to be tested.
            proc_builders, channel_map = self.compiler._compile_proc_groups(
                proc_groups, channel_map
            )

        # There should be six Process Builders...
        self.assertEqual(len(proc_builders), 2)
        # ...one for every Process in the ProcGroups.
        self.assertIsInstance(
            proc_builders[proc_groups[0][0]], PyProcessBuilder
        )
        self.assertIsInstance(
            proc_builders[proc_groups[1][0]], PyProcessBuilder
        )

    def test_compile(self) -> None:
        """Tests the method compile() with a connected network of Processes.
        """

        # Create some Processes and connect them.
        p1, p2, p3, p4, p5 = ProcA(), ProcA(), ProcA(), ProcB(), ProcB()
        p1.out.connect(p2.inp)
        p2.out.connect(p3.inp)
        p3.out.connect(p4.inp)
        p4.out.connect(p5.inp)
        # Create an executable by compiling the Processes.

        executable = self.compiler.compile(p1, RunCfg())

        # There should be five PyProcessBuilders,
        # and four ChannelBuilder in the Executable.
        self.assertEqual(len(executable.proc_builders), 5)
        self.assertEqual(len(executable.channel_builders), 4)

        # Check that the Builders have the correct type.
        for py_builder in executable.proc_builders.values():
            self.assertIsInstance(py_builder, PyProcessBuilder)
        for channel_builder in executable.channel_builders:
            self.assertIsInstance(channel_builder, ChannelBuilderMp)

    def test_create_sync_domain(self):
        """Check creation of custom and default sync domains.

        There are two ways to assign processes to sync domains:
        1. Create custom sync domains an manually assign processes to them
        2. Let compiler assign processes to default sync domains based on the
        sync domain that the chosen process model implements
        """

        # Let's create some processes that implement different sync domains
        p1, p2, p3, p4, p5 = ProcA(), ProcA(), ProcA(), ProcB(), ProcB()
        # Let's also create some processes that does not implement a sync
        # protocol explicitly
        p6, p7 = ProcC(), ProcC()

        # We can create a custom sync domain for each type with one or more
        # processes
        sd_a = SyncDomain(name="sd_a", processes=[p1, p2], protocol=ProtocolA())
        sd_b = SyncDomain(name="sd_b", processes=[p4], protocol=ProtocolB())

        # To compile, with custom sync domains, we pass them to the RunConfig
        run_cfg = RunCfg(custom_sync_domains=[sd_a, sd_b])

        # Here we manually create a proc_group
        proc_group = []
        pm_mapping = {
            p1: PyProcModelA,
            p2: PyProcModelA,
            p3: PyProcModelA,
            p4: PyProcModelB,
            p5: PyProcModelB,
            p6: PyProcModelC,
            p7: PyProcModelC,
        }
        for p, pm in pm_mapping.items():
            p._model_class = pm
            proc_group.append(p)

        # Processes manually assigned so a sync domain will remain there.
        # Processes not assigned to a custom sync domain will get assigned
        # automatically to a default sync domain created for each unique sync
        # protocol
        c = Compiler()
        sd = c._create_sync_domains([proc_group], run_cfg, [], log=c.log)

        # We expect 5 sync domains: The 2 custom ones and 2 default ones
        # created for p3 and p5 and 1 AsyncDomain for processes not
        # implementing a protocol explicitly.
        self.assertEqual(len(sd[0]), 5)
        # The first to domains are sd_a and sd_b
        self.assertEqual(sd[0][0], sd_a)
        self.assertEqual(sd[0][1], sd_b)
        # The next two are auto-generated for the unassigned ProcA and ProcB
        # processes
        self.assertEqual(sd[0][2].name, "ProtocolA_SyncDomain")
        self.assertEqual(sd[0][2].protocol.__class__, ProtocolA)
        self.assertEqual(sd[0][3].name, "ProtocolB_SyncDomain")
        self.assertEqual(sd[0][3].protocol.__class__, ProtocolB)
        # The last one is also auto-generated but in addition,
        # the AsyncProtocol was automatically assigned
        self.assertEqual(sd[0][4].name, "AsyncProtocol_SyncDomain")
        self.assertEqual(sd[0][4].protocol.__class__, AsyncProtocol)

    def test_create_sync_domains_run_config_without_sync_domain(self):
        """Checks that if RunConfig has no SyncDomain, all defaults are
        chosen."""

        # Create two processes and a default RunConfig without explicit
        # SyncDomains
        p1, p2 = ProcA(), ProcB()
        run_cfg = RunCfg()
        p1._model_class = PyProcModelA
        p2._model_class = PyProcModelB
        proc_group = [p1, p2]

        # In this case, only default SyncDomains will be chosen based on the
        # implemented SyncProtocols of each ProcessModel
        c = Compiler()
        sd = c._create_sync_domains([proc_group], run_cfg, [], log=c.log)

        self.assertEqual(len(sd[0]), 2)
        self.assertEqual(sd[0][0].protocol.__class__, ProtocolA)
        self.assertEqual(sd[0][1].protocol.__class__, ProtocolB)

    def test_create_sync_domains_proc_assigned_to_multiple_domains(self):
        """Checks that processes cannot be assigned to multiple sync domains."""

        p1 = ProcA()
        p1._model_class = PyProcModelA
        proc_group = [p1]

        sdx = SyncDomain(name="x", protocol=ProtocolA(), processes=[p1])
        sdy = SyncDomain(name="y", protocol=ProtocolA(), processes=[p1])

        run_cfg = RunCfg(custom_sync_domains=[sdx, sdy])

        c = Compiler()
        with self.assertRaises(AssertionError):
            c._create_sync_domains([proc_group], run_cfg, [], log=c.log)

    def test_create_sync_domains_proc_assigned_to_incompatible_domain(self):
        """Checks that a process can only be assigned to a sync domain with a
        protocol compatible with the sync domain that the chosen process
        model implements."""

        # Create processes implementing ProtocolA and ProtocolB
        p1, p2, p3 = ProcA(), ProcA(), ProcB()
        p1._model_class = PyProcModelA
        p2._model_class = PyProcModelA
        p3._model_class = PyProcModelB
        proc_group = [p1, p2, p3]

        # ...but assign all of them to a domain with ProtocolA
        sd = SyncDomain(name="sd", protocol=ProtocolA(), processes=[p1, p2, p3])
        run_cfg = RunCfg(custom_sync_domains=[sd])

        # In this case, sync domain creation will fail
        c = Compiler()
        with self.assertRaises(AssertionError):
            c._create_sync_domains([proc_group], run_cfg, [], log=c.log)

    def test_create_sync_domain_non_unique_domain_names(self):
        """Checks that sync domain names must be unique."""

        # Let's create two processes...
        p1, p2 = ProcA(), ProcB()
        p1._model_class = PyProcModelA
        p2._model_class = PyProcModelA
        proc_group = [p1, p2]

        # ...that go into two different domains; yet the domains having the
        # same name
        sd1 = SyncDomain(name="sd", protocol=ProtocolA(), processes=[p1])
        sd2 = SyncDomain(name="sd", protocol=ProtocolB(), processes=[p2])
        run_cfg = RunCfg(custom_sync_domains=[sd1, sd2])

        # This does not compile because domain names must be unique
        c = Compiler()
        with self.assertRaises(AssertionError):
            c._create_sync_domains([proc_group], run_cfg, [], log=c.log)

    def test_create_node_cfgs(self):
        """Checks creation of NodeConfigs.

        The implementation of _create_node_cfg is not yet complete. Currently
        we just create a single NodeConfig containing a single Node of type
        'HeadNode' as long as we only support execution on CPU in Python.
        """

        # Let's create some arbitrary processes and a manual proc_group
        p1, p2, p3 = ProcA(), ProcB(), ProcC()
        p1._model_class = PyProcModelA
        p2._model_class = PyProcModelB
        p3._model_class = SubProcModelA
        proc_group = [p1, p2, p3]

        # This creates the naive NodeConfig for now:
        c = Compiler()
        ncfgs = c._create_node_cfgs(proc_groups=[proc_group])

        # It will be a single NodeCfg of type HeadNode containing all processes
        from lava.magma.core.resources import HeadNode

        self.assertEqual(len(ncfgs), 1)
        self.assertEqual(ncfgs[0].nodes[0].node_type, HeadNode)
        # The nodes contain references to all its processes...
        self.assertEqual(ncfgs[0].nodes[0].processes, [p1, p2, p3])
        # ...and the NodeCfg allows to map each process to its node
        head_node = ncfgs[0].nodes[0]
        self.assertEqual(ncfgs[0].node_map[p1], head_node)
        self.assertEqual(ncfgs[0].node_map[p2], head_node)
        self.assertEqual(ncfgs[0].node_map[p3], head_node)

    @unittest.skip("SR: Needs comprehensive refactoring.")
    def test_create_var_models(self):
        """Tests creation of VarModels.

        VarModels map a Var id to a the address information describing where
        and how a Var is implemented in a ProcessModel.
        """

        # Let's create 3 processes but only ProcB-type processes have one Var
        # each. So p1, p2 have one var but p3 has none
        p1, p2, p3 = ProcB(), ProcB(), ProcC()

        # Create proc_group manually
        p1._model_class = PyProcModelB
        p2._model_class = PyProcModelB
        p3._model_class = PyProcModelC
        proc_groups = [[p1, p2, p3]]

        # First we need to compile a NodeConfig
        c = Compiler()
        node_cfgs = c._create_node_cfgs(proc_groups)

        # Creating exec_vars adds any ExecVars to each NodeConfig
        c._assign_nodecfg_rtservice_to_var_models(
            node_cfgs, proc_groups, {p1.id: 0, p2.id: 0, p3.id: 0}
        )
        var_models = node_cfgs[0].var_models

        # Since only p1 and p2 have Vars, there will be 2 ExecVars
        self.assertEqual(len(var_models), 2)

        # exec_vars is a map from Var.id to ExecVar
        ev1 = var_models[p1.some_var.id]
        ev2 = var_models[p2.some_var.id]

        # Both Vars have the same name in each Process
        self.assertEqual(ev1.name, "some_var")
        self.assertEqual(ev2.name, "some_var")
        # ...and the same ids as their corresponding Vars
        self.assertEqual(ev1.var_id, p1.some_var.id)
        self.assertEqual(ev2.var_id, p2.some_var.id)
        # ...and both are on the same node and same runtime service
        self.assertEqual(ev1.node_id, 0)
        self.assertEqual(ev2.node_id, 0)
        self.assertEqual(ev1.runtime_srv_id, 0)
        self.assertEqual(ev2.runtime_srv_id, 0)

    @unittest.skip("MR: Skipping for merge.")
    def test_compile_fail_on_multiple_compilation(self):
        """Checks that a process can only be compiled once."""

        # Create multiple top level processes; one with sub process structure
        p1 = ProcA()
        p2 = ProcB()
        p1.out.connect(p2.inp)

        # Compile processes once
        c = Compiler()
        c.compile(process=p1, run_cfg=RunCfg(select_sub_proc_model=True))

        # As a result of compilation, the _is_compiled flag has been set for
        # top level processes
        self.assertTrue(p1._is_compiled)
        self.assertTrue(p2._is_compiled)
        self.assertFalse(p1.procs.proc1._is_compiled)
        self.assertFalse(p1.procs.proc2._is_compiled)

        # Afterwards, compiling any of the same processes again must fail
        with self.assertRaises(ex.ProcessAlreadyCompiled):
            c.compile(process=p2, run_cfg=RunCfg(select_sub_proc_model=True))


if __name__ == "__main__":
    unittest.main()
