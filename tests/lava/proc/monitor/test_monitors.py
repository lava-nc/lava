import numpy as np
import unittest

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyRefPort, PyVarPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import RefPort, VarPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.decorator import implements, requires
from lava.proc.monitor.process import Monitor
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.run_configs import Loihi1SimCfg


class P1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.v = Var(shape=(1,), init=1)
        self.u = Var(shape=(1,), init=0)


# A minimal PyProcModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
class PyProcModel1(PyLoihiProcessModel):
    v: np.ndarray = LavaPyType(np.ndarray, np.int32)
    u: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def pre_guard(self):
        return True

    def run_pre_mgmt(self):
        if self.current_ts > 1:
            self.v = np.array([self.current_ts])
            self.u = 2*np.array([self.current_ts])


class Monitors(unittest.TestCase):
    def test_monitor_constructor(self):
        monitor = Monitor()

        self.assertIsInstance(monitor, Monitor)

        # simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
        #                                 [some_proc, monitor])
        # Dynamically create refPorts for each var to be monitored rather
        # than hard-coding the refports

    def test_monitor_add_probe_create_ref_var_ports(self):
        num_steps = 4
        monitor = Monitor()
        some_proc = P1()
        monitor.probe(var=some_proc.v, num_steps=num_steps)

    def test_monitor_add_probe_create_connection(self):
        num_steps = 4
        monitor = Monitor()
        some_proc = P1()
        monitor.probe(var=some_proc.v, num_steps=num_steps)

        # Regardless where we start searching...
        c = Compiler()
        procs1 = c._find_processes(some_proc)
        procs2 = c._find_processes(monitor)

        # ...we will find all of them
        all_procs = {some_proc, monitor}
        self.assertEqual(set(procs1), all_procs)
        self.assertEqual(set(procs2), all_procs)

    def test_monitor_and_proc_run_without_error(self):
        num_steps = 4
        monitor = Monitor()
        some_proc = P1()
        monitor.probe(var=some_proc.v, num_steps=num_steps)

        class MyRunCfg(RunConfig):
            def select(self, proc, proc_models):
                return proc_models[0]

        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [some_proc, monitor])

        # should run without error (not doing anything)
        some_proc.run(RunSteps(num_steps=num_steps, blocking=True),
                      MyRunCfg(custom_sync_domains=[simple_sync_domain]))

        some_proc.stop()

    # def test_monitor_probe_created_empty_data_collection_structure(self):
    #     monitor = Monitor()
    #     some_proc = P1()
    #     monitor.probe(var=some_proc.v)
    #
    #     self.assertEqual(monitor.data[some_proc.name][some_proc.v.name], [])

    def test_monitor_collects_correct_data_from_one_var(self):
        num_steps = 6
        monitor = Monitor()
        some_proc = P1()
        monitor.probe(var=some_proc.v, num_steps=num_steps)

        class MyRunCfg(RunConfig):
            def select(self, proc, proc_models):
                return proc_models[0]

        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [some_proc, monitor])

        # should run without error (not doing anything)
        some_proc.run(RunSteps(num_steps=num_steps, blocking=True),
                      MyRunCfg(custom_sync_domains=[simple_sync_domain]))

        print(monitor.var_read.get())
        self.assertTrue(
            np.all(monitor.var_read.get() == np.array([1, 2, 3, 4, 5, 6])))

        some_proc.stop()
        # self.assertEqual(monitor.data[some_proc.name][some_proc.v.name],
        #                  np.array([1, 2, 3, 4]))

    def test_monitor_collects_correct_data_from_two_vars(self):
        num_steps = 4
        monitor = Monitor()
        some_proc = P1()
        monitor.probe(var=some_proc.v, num_steps=num_steps)
        monitor.probe(var=some_proc.u, num_steps=num_steps)
        print("a")
        # class MyRunCfg(RunConfig):
        #     def select(self, proc, proc_models):
        #         return proc_models[0]
        #
        # simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
        #                                 [some_proc, monitor])
        #
        # # should run without error (not doing anything)
        # some_proc.run(RunSteps(num_steps=4, blocking=True),
        #               MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        #
        # self.assertTrue(
        #         np.all(monitor.var_read.get() == np.array([1, 2, 3, 4])))
        # some_proc.stop()

    def test_proc_params_accessible_in_proc_model(self):
        num_steps = 4
        monitor = Monitor()
        some_proc = P1()
        monitor.probe(var=some_proc.v, num_steps=num_steps)
        monitor.proc_params = {"test": 0}

        class MyRunCfg(RunConfig):
            def select(self, proc, proc_models):
                return proc_models[0]

        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [some_proc, monitor])

        # Regardless where we start searching...
        c = Compiler()
        exe = c.compile(some_proc,
                        MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        self.assertEqual(next(iter(exe.py_builders)).proc_params,
                         monitor.proc_params)
        print("done")


if __name__ == '__main__':
    unittest.main()
