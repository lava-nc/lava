# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import logging
import unittest

from lava.magma.compiler.channels.interfaces import ChannelType
from lava.magma.compiler.compiler import Compiler
import lava.magma.compiler.exceptions as ex
from lava.magma.core.decorator import implements, requires
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.reduce_ops import ReduceSum
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.process.ports.ports import (
    InPort, OutPort, RefPort, VarPort)
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort, \
    PyVarPort
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.variable import Var, VarServer
from lava.magma.core.resources import CPU
from lava.magma.core.process.ports.ports import create_port_id


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
    pass


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
    def __init__(self,
                 loglevel: int = logging.WARNING,
                 custom_sync_domains=None,
                 select_sub_proc_model=False):
        super().__init__(custom_sync_domains=custom_sync_domains,
                         loglevel=loglevel)
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


class TestCompiler(unittest.TestCase):
    """The compiler consumes a single Process, discovers all connected
    Processes and continues to compile all of them into an Executable is in
    turn consumed by the Runtime.

    Here we first walk through all internal compiler methods and test them
    separately. Only in the end, do we perform a complete compilation from
    Process to Executable.
    """

    def setUp(self):
        VarServer().reset_server()

    def test_constructor(self):
        """Checks compiler can be instantiated."""

        compiler = Compiler()
        self.assertIsInstance(compiler, Compiler)

    def test_find_processes_disconnected(self):
        """Checks finding of processes. Disconnected processes will not be
        found."""

        # Create two disconnected processes
        proc1 = ProcA()
        proc2 = ProcA()

        # When trying to find processes starting from lava.proc1 ...
        c = Compiler()
        procs = c._find_processes(proc1)

        # ... only proc1 will be found
        self.assertEqual(procs, [proc1])
        self.assertFalse(proc2 in procs)

    def test_find_processes_connected(self):
        """Check finding all processes in a linear sequence of connections."""

        # Create three processes to be connected
        proc1 = ProcA()
        proc2 = ProcA()
        proc3 = ProcA()

        # This time, we connect them in a sequence
        proc1.out.connect(proc2.inp)
        proc2.out.connect(proc3.inp)

        # Regardless where we start searching...
        c = Compiler()
        procs1 = c._find_processes(proc1)
        procs2 = c._find_processes(proc2)
        procs3 = c._find_processes(proc3)

        # ... we will find all of them (although not necessarily in same order)
        all_procs = {proc1, proc2, proc3}
        self.assertEqual(set(procs1), all_procs)
        self.assertEqual(set(procs2), all_procs)
        self.assertEqual(set(procs3), all_procs)

    def test_find_process_joining_forking(self):
        """Checks finding all processes for joining and forking connections:
        [(p1 -> p2), p3] -> p4 -> p5 -> [p6, (p7 -> p8)] -> p9"""

        # Create processes
        p1, p2, p3, p4, p5 = ProcA(), ProcA(), ProcA(), ProcA(), ProcA()
        p6, p7, p8, p9 = ProcA(), ProcA(), ProcA(), ProcA()

        # Create complicated joining and branching structure
        p1.out.connect(p2.inp)
        # (Make use of convenient connect_from to make two fan-in connections)
        p4.inp.connect_from([p2.out, p3.out])
        p4.out.connect(p5.inp)
        p5.out.connect([p6.inp, p7.inp])
        p7.out.connect(p8.inp)
        # (Manually create two individual fan-in connections just because...)
        p6.out.connect(p9.inp)
        p8.out.connect(p9.inp)

        # Regardless where we start searching (p4, p5 are equivalent)...
        c = Compiler()
        procs1 = c._find_processes(p1)
        procs5 = c._find_processes(p5)
        procs9 = c._find_processes(p9)

        # ...we will find all of them
        all_procs = {p1, p2, p3, p4, p5, p6, p7, p8, p9}
        self.assertEqual(set(procs1), all_procs)
        self.assertEqual(set(procs5), all_procs)
        self.assertEqual(set(procs9), all_procs)

    def test_find_process_circular(self):
        """Checks finding all processes for circular connections.
        (p1, p2) -> p3 -> [(p4 -> p1), (p5 -> p6 -> p3)]"""

        # Create processes
        p1, p2, p3, p4, p5, p6 = (
            ProcA(),
            ProcA(),
            ProcA(),
            ProcA(),
            ProcA(),
            ProcA(),
        )
        # Create complicated circular structure with joins and forks
        p3.inp.connect_from([p1.out, p2.out])
        p3.out.connect([p4.inp, p5.inp])
        p4.out.connect(p1.inp)
        p5.out.connect(p6.inp)
        p6.out.connect(p3.inp)

        # Regardless where we start searching...
        c = Compiler()
        procs2 = c._find_processes(p2)
        procs4 = c._find_processes(p4)
        procs6 = c._find_processes(p6)

        # ...we will find all of them
        all_procs = {p1, p2, p3, p4, p5, p6}
        self.assertEqual(set(procs2), all_procs)
        self.assertEqual(set(procs4), all_procs)
        self.assertEqual(set(procs6), all_procs)

    def test_find_process_ref_ports(self):
        """Checks finding all processes for RefPort connection.
        [p1 -> ref/var -> p2 -> out/in -> p3]"""

        # Create processes
        p1, p2, p3 = ProcA(), ProcB(), ProcC()

        # Connect p1 (RefPort) with p2 (VarPort)
        p1.ref.connect(p2.var_port)
        # Connect p2 (OutPort) with p3 (InPort)
        p2.out.connect(p3.inp)

        # Regardless where we start searching...
        c = Compiler()
        procs1 = c._find_processes(p1)
        procs2 = c._find_processes(p2)
        procs3 = c._find_processes(p3)

        # ...we will find all of them
        all_procs = {p1, p2, p3}
        self.assertEqual(set(procs1), all_procs)
        self.assertEqual(set(procs2), all_procs)
        self.assertEqual(set(procs3), all_procs)

    def test_find_processes_across_virtual_ports(self):
        """Tests whether in Process graphs with virtual ports, all Processes
        are found, no matter from which Process the search is started."""

        source = ProcC()
        sink = ProcC()
        source.out.reshape((1,)).reshape((1,)).connect(sink.inp)

        compiler = Compiler()
        # Test whether all Processes are found when starting the search from
        # the source Process
        expected_procs = [sink, source]
        found_procs = compiler._find_processes(source)
        self.assertCountEqual(found_procs, expected_procs)

        # Test whether all Processes are found when starting the search from
        # the destination Process
        found_procs = compiler._find_processes(sink)
        self.assertCountEqual(found_procs, expected_procs)

    def test_find_proc_models(self):
        """Check finding of ProcModels that implement a Process."""

        # Create a Process for which we want to find ProcModels
        proc = ProcA()

        # Find ProcessModels
        c = Compiler()
        proc_models = c._find_proc_models(proc)

        # This will the two ProcModels defined in this Python module that
        # implement Proc
        self.assertEqual(len(proc_models), 2)
        self.assertEqual(proc_models, [PyProcModelA, SubProcModelA])

    def test_find_proc_models_failing(self):
        """Checks that search for ProcessModels fails if no ProcessModels are
        found that implement a Process."""

        class AnotherProc(AbstractProcess):
            pass

        proc = AnotherProc()

        # Since there are not ProcModels that implement AnotherProc,
        # the search will fail
        c = Compiler()
        with self.assertRaises(ex.NoProcessModelFound):
            c._find_proc_models(proc)

    def test_map_proc_to_model_no_sub_proc(self):
        """Checks generation of map from lava.process to its selected
        ProcessModel without SubProcModels."""

        # Create three processes and connect them in linear sequence
        p1 = ProcA()
        p2 = ProcB()
        p3 = ProcA()
        p1.out.connect(p2.inp)
        p2.out.connect(p3.inp)

        # First we can find all processes (although we already know them)
        c = Compiler()
        procs = c._find_processes(p1)

        # In order to map Processes to they ProcessModels we need a RunCfg.
        # How the RunCfg works is not important and out of scope for this
        # unit test. Here we just always select the direct PyProcModel by
        # disallowing to select SubProcModels.
        run_cfg = RunCfg(select_sub_proc_model=False)

        # Then we can get the ProcessModels
        proc_map = c._map_proc_to_model(procs, run_cfg)

        # All Processes have been mapped to PyProcModels directly
        self.assertEqual(proc_map[p1], PyProcModelA)
        self.assertEqual(proc_map[p2], PyProcModelB)
        self.assertEqual(proc_map[p3], PyProcModelA)

    def test_map_proc_to_model_with_sub_proc(self):
        """Checks generation of map from lava.process to its selected
        ProcessModel with SubProcModels."""

        # Create three processes and connect them in linear sequence
        p1 = ProcA()
        p2 = ProcB()
        p3 = ProcA()
        p1.out.connect(p2.inp)
        p2.out.connect(p3.inp)

        # First we can find all processes (although we already know them)
        c = Compiler()
        procs = c._find_processes(p1)

        # This time we allow expansion of sub processes
        run_cfg = RunCfg(select_sub_proc_model=True)

        # Then we can get the ProcessModels
        proc_map = c._map_proc_to_model(procs, run_cfg)

        # p1::ProcA and p3::ProcA have been expanded to ProcB and ProcC while
        # ProcB and ProcC have been mapped to their corresponding PyProcModels
        self.assertFalse(p1.procs.is_empty())
        self.assertEqual(proc_map[p1.procs.proc1], PyProcModelB)
        self.assertEqual(proc_map[p1.procs.proc2], PyProcModelC)

        self.assertTrue(p2.procs.is_empty())
        self.assertEqual(proc_map[p2], PyProcModelB)

        self.assertFalse(p3.procs.is_empty())
        self.assertEqual(proc_map[p3.procs.proc1], PyProcModelB)
        self.assertEqual(proc_map[p3.procs.proc2], PyProcModelC)

    def test_group_proc_by_model(self):
        """Checks grouping of Processes by the ProcessModels implementing it."""

        # Let's a set of processes (for the purpose of this test they do not
        # need to be connected)
        p1 = ProcA()
        p2 = ProcC()
        p3 = ProcA()
        p4 = ProcB()

        # Normally the compiler would create a map from lava.process to
        # ProcessModel
        proc_map = {
            p1: PyProcModelA,
            p2: PyProcModelC,
            p3: PyProcModelA,
            p4: PyProcModelB,
        }

        # Then the Compiler would group Processes by their ProcessModel
        c = Compiler()
        groups = c._group_proc_by_model(proc_map)

        # ProcA will map to p1 and p3 while the others map only to one process
        self.assertEqual(groups[PyProcModelA], [p1, p3])
        self.assertEqual(groups[PyProcModelB], [p4])
        self.assertEqual(groups[PyProcModelC], [p2])

    def test_compile_py_proc_models(self):
        """Checks compilation of ProcessModels which (in this example) only
        generated PyProcBuilders for each Process and ProcessModel."""

        # Normally, the compiler would generate proc_groups which maps a
        # ProcessModel type to a list of processes implemented by it
        p1, p2, p3 = ProcA(), ProcA(), ProcB()
        proc_groups = {PyProcModelA: [p1, p2], PyProcModelB: [p3]}

        # Compiling these proc_groups will return an Executable initialized
        # with PyProcBuilders
        c = Compiler()
        exe = c._compile_proc_models(proc_groups)

        # The executable should have 3 PyProcBuilders...
        self.assertEqual(len(exe.py_builders), 3)
        # ... one for each Process
        b1 = exe.py_builders[p1]
        b2 = exe.py_builders[p2]
        b3 = exe.py_builders[p3]
        self.assertEqual(b1.proc_model, PyProcModelA)
        self.assertEqual(b2.proc_model, PyProcModelA)
        self.assertEqual(b3.proc_model, PyProcModelB)

        # ProcA builders only have PortInitializers for 'inp' and 'out' while
        # ProcB has also a VarInitializer
        self.assertEqual(len(b1.vars), 0)
        self.assertEqual(b1.py_ports["inp"].name, "inp")
        self.assertEqual(b1.py_ports["inp"].shape, (1,))
        self.assertEqual(b1.py_ports["out"].name, "out")

        self.assertEqual(len(b3.vars), 1)
        self.assertEqual(b3.vars["some_var"].name, "some_var")
        self.assertEqual(b3.vars["some_var"].value, 10)
        self.assertEqual(b3.py_ports["inp"].name, "inp")

    def test_compile_py_proc_models_with_virtual_ports(self):
        """Checks compilation of ProcessModels when Processes are connected
        via virtual ports."""

        # Normally, the compiler would generate proc_groups which maps a
        # ProcessModel type to a list of processes implemented by it
        p1, p2, p3 = ProcA(), ProcA(), ProcB()
        p1.out.flatten().connect(p3.inp)
        p2.ref.flatten().connect(p3.var_port)
        proc_groups = {PyProcModelA: [p1, p2], PyProcModelB: [p3]}

        # Compiling these proc_groups will return an Executable initialized
        # with PyProcBuilders
        c = Compiler()
        exe = c._compile_proc_models(proc_groups)

        # Get the PyProcBuilder for Processes that have PyPorts that may
        # receive data
        b2 = exe.py_builders[p2]
        b3 = exe.py_builders[p3]

        # Check whether the transformation functions are registered in the
        # PortInitializers
        self.assertEqual(list(b3.py_ports["inp"].transform_funcs.keys()),
                         [create_port_id(p1.id, p1.out.name)])
        self.assertEqual(list(b3.var_ports["var_port"].transform_funcs.keys()),
                         [create_port_id(p2.id, p2.ref.name)])
        self.assertEqual(list(b2.ref_ports["ref"].transform_funcs.keys()),
                         [create_port_id(p3.id, p3.var_port.name)])

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
        sd_a = SyncDomain(name="sd_a", processes=[
                          p1, p2], protocol=ProtocolA())
        sd_b = SyncDomain(name="sd_b", processes=[p4], protocol=ProtocolB())

        # To compile, with custom sync domains, we pass them to the RunConfig
        run_cfg = RunCfg(custom_sync_domains=[sd_a, sd_b])

        # Here we manually create a proc_map
        proc_map = {
            p1: PyProcModelA,
            p2: PyProcModelA,
            p3: PyProcModelA,
            p4: PyProcModelB,
            p5: PyProcModelB,
            p6: PyProcModelC,
            p7: PyProcModelC,
        }

        # Processes manually assigned so a sync domain will remain there.
        # Processes not assigned to a custom sync domain will get assigned
        # automatically to a default sync domain created for each unique sync
        # protocol
        c = Compiler()
        sd = c._create_sync_domains(proc_map, run_cfg,
                                    [],
                                    log=c.log)

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
        proc_map = {p1: PyProcModelA, p2: PyProcModelB}

        # In this case, only default SyncDomains will be chosen based on the
        # implemented SyncProtocols of each ProcessModel
        c = Compiler()
        sd = c._create_sync_domains(proc_map,
                                    run_cfg,
                                    [],
                                    log=c.log)

        self.assertEqual(len(sd[0]), 2)
        self.assertEqual(sd[0][0].protocol.__class__, ProtocolA)
        self.assertEqual(sd[0][1].protocol.__class__, ProtocolB)

    def test_create_sync_domains_proc_assigned_to_multiple_domains(self):
        """Checks that processes cannot be assigned to multiple sync domains."""

        p1 = ProcA()

        sdx = SyncDomain(name="x", protocol=ProtocolA(), processes=[p1])
        sdy = SyncDomain(name="y", protocol=ProtocolA(), processes=[p1])

        run_cfg = RunCfg(custom_sync_domains=[sdx, sdy])

        proc_map = {p1: PyProcModelA}

        c = Compiler()
        with self.assertRaises(AssertionError):
            c._create_sync_domains(proc_map,
                                   run_cfg,
                                   [],
                                   log=c.log)

    def test_create_sync_domains_proc_assigned_to_incompatible_domain(self):
        """Checks that a process can only be assigned to a sync domain with a
        protocol compatible with the sync domain that the chosen process
        model implements."""

        # Create processes implementing ProtocolA and ProtocolB
        p1, p2, p3 = ProcA(), ProcA(), ProcB()

        # ...but assign all of them to a domain with ProtocolA
        sd = SyncDomain(name="sd", protocol=ProtocolA(),
                        processes=[p1, p2, p3])
        run_cfg = RunCfg(custom_sync_domains=[sd])
        proc_map = {p1: PyProcModelA, p2: PyProcModelA, p3: PyProcModelB}

        # In this case, sync domain creation will fail
        c = Compiler()
        with self.assertRaises(AssertionError):
            c._create_sync_domains(proc_map,
                                   run_cfg,
                                   [],
                                   log=c.log)

    def test_create_sync_domain_non_unique_domain_names(self):
        """Checks that sync domain names must be unique."""

        # Let's create two processes...
        p1, p2 = ProcA(), ProcB()

        # ...that go into two different domains; yet the domains having the
        # same name
        sd1 = SyncDomain(name="sd", protocol=ProtocolA(), processes=[p1])
        sd2 = SyncDomain(name="sd", protocol=ProtocolB(), processes=[p2])
        run_cfg = RunCfg(custom_sync_domains=[sd1, sd2])
        proc_map = {p1: PyProcModelA, p2: PyProcModelB}

        # This does not compile because domain names must be unique
        c = Compiler()
        with self.assertRaises(AssertionError):
            c._create_sync_domains(proc_map,
                                   run_cfg,
                                   [],
                                   log=c.log)

    def test_create_node_cfgs(self):
        """Checks creation of NodeConfigs.

        The implementation of _create_node_cfg is not yet complete. Currently
        we just create a single NodeConfig containing a single Node of type
        'HeadNode' as long as we only support execution on CPU in Python.
        """

        # Let's create some arbitrary processes and a manual proc_map
        p1, p2, p3 = ProcA(), ProcB(), ProcC()
        proc_map = {p1: PyProcModelA, p2: PyProcModelB, p3: SubProcModelA}

        # This creates the naive NodeConfig for now:
        c = Compiler()
        ncfgs = c._create_node_cfgs(proc_map, log=c.log)

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

        # Create a manual proc_map
        proc_map = {
            p1: PyProcModelA,
            p2: PyProcModelB,
            p3: PyProcModelC,
            p4: PyProcModelA,
            p5: PyProcModelB,
        }

        # Create channel builders
        c = Compiler()
        cbs = c._create_channel_builders(proc_map)

        # This should result in 5 channel builders (one for each arrow above)
        from lava.magma.compiler.builders.builder import ChannelBuilderMp

        self.assertEqual(len(cbs), 5)
        for cb in cbs:
            self.assertIsInstance(cb, ChannelBuilderMp)

        # Each channel builder should connect its source and destination
        # process and port
        # Let's check the first one in detail
        self.assertEqual(cbs[0].channel_type, ChannelType.PyPy)
        self.assertEqual(cbs[0].src_process, p1)
        self.assertEqual(cbs[0].src_port_initializer.name, "out")
        self.assertEqual(cbs[0].src_port_initializer.shape, (1,))
        self.assertEqual(cbs[0].src_port_initializer.d_type, int)
        self.assertEqual(cbs[0].src_port_initializer.size, 64)
        self.assertEqual(cbs[0].dst_process, p2)
        # ... and for others only the source/destination processes
        self.assertEqual(cbs[1].src_process, p2)
        self.assertEqual(cbs[1].dst_process, p3)
        self.assertEqual(cbs[2].src_process, p3)
        self.assertEqual(cbs[2].dst_process, p4)
        self.assertEqual(cbs[3].src_process, p4)
        self.assertEqual(cbs[3].dst_process, p2)
        self.assertEqual(cbs[4].src_process, p4)
        self.assertEqual(cbs[4].dst_process, p5)

    def test_create_channel_builders_hierarchical_process(self):
        """Check creation of channel builders when a SubProcModel is used
        with connections potentially terminating in dangling ports of
        hierarchical processes."""

        # Create a minimal process
        p = ProcA()

        # Manually map processes to sub processes to internal sub structure
        # gets build
        c = Compiler()
        proc_map = c._map_proc_to_model([p],
                                        RunCfg(select_sub_proc_model=True))

        chb = c._create_channel_builders(proc_map)

        # There should only be one ChannelBuilder from the internal proc1 to
        # proc2
        self.assertEqual(len(chb), 1)
        from lava.magma.compiler.builders.builder import ChannelBuilderMp
        self.assertIsInstance(chb[0], ChannelBuilderMp)
        self.assertEqual(chb[0].src_process, p.procs.proc1)
        self.assertEqual(chb[0].dst_process, p.procs.proc2)

    def test_create_channel_builders_ref_ports(self):
        """Checks creation of channel builders when a process is connected
        using a RefPort to another process VarPort."""

        # Create a process with a RefPort (source)
        src = ProcA()

        # Create a process with a var (destination)
        dst = ProcB()

        # Connect them using RefPort and VarPort
        src.ref.connect(dst.var_port)

        # Create a manual proc_map
        proc_map = {
            src: PyProcModelA,
            dst: PyProcModelB
        }

        # Create channel builders
        c = Compiler()
        cbs = c._create_channel_builders(proc_map)

        # This should result in 2 channel builder
        from lava.magma.compiler.builders.builder import ChannelBuilderMp
        self.assertEqual(len(cbs), 2)
        self.assertIsInstance(cbs[0], ChannelBuilderMp)
        self.assertEqual(cbs[0].src_process, src)
        self.assertEqual(cbs[0].dst_process, dst)

    def test_create_channel_builders_ref_ports_implicit(self):
        """Checks creation of channel builders when a process is connected
        using a RefPort to another process Var (implicit VarPort)."""

        # Create a process with a RefPort (source)
        src = ProcA()

        # Create a process with a var (destination)
        dst = ProcB()

        # Connect them using RefPort and Var (creates implicitly a VarPort)
        src.ref.connect_var(dst.some_var)

        # Create a manual proc_map
        proc_map = {
            src: PyProcModelA,
            dst: PyProcModelB
        }

        # Create channel builders
        c = Compiler()
        cbs = c._create_channel_builders(proc_map)

        # This should result in 2 channel builder
        from lava.magma.compiler.builders.builder import ChannelBuilderMp
        self.assertEqual(len(cbs), 2)
        self.assertIsInstance(cbs[0], ChannelBuilderMp)
        self.assertEqual(cbs[0].src_process, src)
        self.assertEqual(cbs[0].dst_process, dst)

    # ToDo: (AW) @YS/@JM Please fix unit test by passing run_srv_builders to
    #  _create_exec_vars when ready
    def test_create_py_exec_vars(self):
        """Tests creation of PyExecVars.

        ExecVars map a Var id to a the address information describing where
        and how a Var is implemented in a ProcessModel.
        """

        # Let's create 3 processes but only ProcB-type processes have one Var
        # each. So p1, p2 have one var but p3 has none
        p1, p2, p3 = ProcB(), ProcB(), ProcC()

        # Create proc_map manually
        proc_map = {p1: PyProcModelA, p2: PyProcModelB, p3: PyProcModelC}

        # First we need to compile a NodeConfig
        c = Compiler()
        node_cfgs = c._create_node_cfgs(proc_map, log=c.log)

        # Creating exec_vars adds any ExecVars to each NodeConfig
        c._create_exec_vars(node_cfgs, proc_map, {
                            p1.id: 0, p2.id: 0, p3.id: 0})
        exec_vars = node_cfgs[0].exec_vars

        # Since only p1 and p2 have Vars, there will be 2 ExecVars
        self.assertEqual(len(exec_vars), 2)

        # exec_vars is a map from Var.id to ExecVar
        ev1 = exec_vars[p1.some_var.id]
        ev2 = exec_vars[p2.some_var.id]

        # Both Vars have the same name in each Process
        self.assertEqual(ev1.name, 'some_var')
        self.assertEqual(ev2.name, 'some_var')
        # ...and the same ids as their corresponding Vars
        self.assertEqual(ev1.var_id, 0)
        self.assertEqual(ev2.var_id, 1)
        # ...and both are on the same node and same runtime service
        self.assertEqual(ev1.node_id, 0)
        self.assertEqual(ev2.node_id, 0)
        self.assertEqual(ev1.runtime_srv_id, 0)
        self.assertEqual(ev2.runtime_srv_id, 0)

    def test_compile_fail_on_multiple_compilation(self):
        """Checks that a process can only be compiled once."""

        # Create multiple top level processes; one with sub process structure
        p1 = ProcA()
        p2 = ProcB()
        p1.out.connect(p2.inp)

        # Compile processes once
        c = Compiler()
        c.compile(proc=p1, run_cfg=RunCfg(select_sub_proc_model=True))

        # As a result of compilation, the _is_compiled flag has been set for
        # top level processes
        self.assertTrue(p1._is_compiled)
        self.assertTrue(p2._is_compiled)
        self.assertFalse(p1.procs.proc1._is_compiled)
        self.assertFalse(p1.procs.proc2._is_compiled)

        # Afterwards, compiling any of the same processes again must fail
        with self.assertRaises(ex.ProcessAlreadyCompiled):
            c.compile(proc=p2, run_cfg=RunCfg(select_sub_proc_model=True))


if __name__ == "__main__":
    unittest.main()
