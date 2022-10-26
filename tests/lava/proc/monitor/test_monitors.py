# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.decorator import implements, requires
from lava.proc.monitor.process import Monitor
from lava.proc.lif.process import LIF
from lava.magma.compiler.compiler import Compiler
from lava.magma.compiler import compiler_graphs
from lava.magma.core.run_configs import Loihi1SimCfg


class P1(AbstractProcess):
    """A dummy proc with two Vars"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.s = Var(shape=(1,), init=1)
        self.u = Var(shape=(1,), init=0)
        self.v = Var(shape=(2, 2), init=np.array([[1, 2], [3, 4]]))


@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
class PyProcModel1(PyLoihiProcessModel):
    """A minimal PyProcModel implementing P1"""

    s: np.ndarray = LavaPyType(np.ndarray, np.int32)
    u: np.ndarray = LavaPyType(np.ndarray, np.int32)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        if self.time_step > 1:
            self.s = np.array([self.time_step])
            self.u = 2 * np.array([self.time_step])
            self.v = np.array([[1, 2], [3, 4]])


class Monitors(unittest.TestCase):

    def test_monitor_constructor(self):
        """Check if a Monitor process is correctly instantiated"""
        monitor = Monitor()
        # check if correct instance is created
        self.assertIsInstance(monitor, Monitor)

    def test_proc_compiled_without_error(self):
        """Check if Monitor Proc is compiled without an error"""
        monitor = Monitor()
        c = Compiler()
        # Compiling should run without error
        c.compile(monitor, Loihi1SimCfg())

    def test_monitor_add_probe_create_ref_var_ports(self):
        """Check if probe(..) method of Monitor Proc creates a new RefPort
        to facilitate the monitoring"""
        num_steps = 4
        # Create a Monitor Process and a dummy process to be probed
        monitor = Monitor()
        some_proc = P1()

        # Probing a Var with Monitor Process
        monitor.probe(target=some_proc.s, num_steps=num_steps)

        # Check if a new RefPort created
        self.assertIsInstance(monitor.ref_ports.members[0], RefPort)

    def test_probing_input_port_raise_error(self):
        """Check if trying to monitor an InPort raise an error"""
        monitor = Monitor()
        # Create a LIF neuron which has an InPort called a_in
        neuron = LIF(shape=(1,),
                     vth=200,
                     bias_mant=5)

        # Check if the type error is raised
        with self.assertRaises(TypeError):
            monitor.probe(target=neuron.a_in, num_steps=2)

    def test_monitor_adding_probe_create_connection(self):
        """ Check if probe(..) actually makes the connection between Monitor
        proc and the dummy proc to be monitored"""
        num_steps = 4
        # create the procs
        monitor = Monitor()
        some_proc = P1()

        # Probing a Var with Monitor Process
        monitor.probe(target=some_proc.s, num_steps=num_steps)

        # Regardless where we start searching...
        procs1 = compiler_graphs.find_processes(some_proc)
        procs2 = compiler_graphs.find_processes(monitor)

        # ...we will find all of the processes
        all_procs = {some_proc, monitor}
        self.assertEqual(set(procs1), all_procs)
        self.assertEqual(set(procs2), all_procs)

    def test_monitor_and_proc_run_without_error(self):
        """Check if the procs run after being probed by Monitor proc"""
        num_steps = 4
        monitor = Monitor()
        some_proc = P1()
        monitor.probe(target=some_proc.s, num_steps=num_steps)

        # Should run without error (not doing anything)
        some_proc.run(condition=RunSteps(num_steps=num_steps),
                      run_cfg=Loihi1SimCfg())

        some_proc.stop()

    def test_monitor_with_2D_var(self):
        """Check if the procs run after a 2-D var being probed by Monitor
        proc without any error"""
        num_steps = 4
        monitor = Monitor()
        some_proc = P1()
        monitor.probe(target=some_proc.v, num_steps=num_steps)

        # Should run without error (not doing anything)
        some_proc.run(condition=RunSteps(num_steps=num_steps),
                      run_cfg=Loihi1SimCfg())

        some_proc.stop()

    def test_monitor_collects_correct_data_from_one_var(self):
        """Check if the collected data in Monitor process matches the
        expected data"""
        num_steps = 6
        monitor = Monitor()
        some_proc = P1()
        monitor.probe(target=some_proc.s, num_steps=num_steps)

        # Run all connected processes
        some_proc.run(condition=RunSteps(num_steps=num_steps),
                      run_cfg=Loihi1SimCfg())

        # Fetch and construct the monitored data with get_data(..) method
        data = monitor.get_data()

        # Access the collected data with the names of monitor proc and var
        probe_data = data[some_proc.name][some_proc.s.name]

        # Check if the collected data match the expected data
        self.assertTrue(np.all(probe_data == np.array([[1, 2, 3, 4, 5, 6]]).T))

        # Stop running
        some_proc.stop()

    def test_monitor_collects_correct_data_from_2D_var(self):
        """Check if the collected data in Monitor process matches the
        expected data"""
        num_steps = 2
        monitor = Monitor()
        some_proc = P1()
        monitor.probe(target=some_proc.v, num_steps=num_steps)

        # Run all connected processes
        some_proc.run(condition=RunSteps(num_steps=num_steps),
                      run_cfg=Loihi1SimCfg())

        # Fetch and construct the monitored data with get_data(..) method
        data = monitor.get_data()

        # Access the collected data with the names of monitor proc and var
        probe_data = data[some_proc.name][some_proc.v.name]

        # Check if the collected data match the expected data
        self.assertTrue(np.all(probe_data == np.tile(np.array([[1, 2], [3, 4]]),
                                                     (num_steps, 1, 1))))

        # Stop running
        some_proc.stop()

    def test_monitor_collects_voltage_and_spike_data_from_lif_neuron(self):
        """Check if two different Monitor process can monitor voltage (Var) and
        s_out (OutPort) of a LIF neuron. Check the collected data with
        expected data.
        Note: The LIF neuron integrate the given bias, voltage accumulates
        and once pass the threshold, there will be a spike outputted and
        voltage will be reset to zero.
        """

        # Setup two Monitor Processes and LIF Process (a single neuron) that
        # will be monitored
        monitor1 = Monitor()
        monitor2 = Monitor()
        shape = (1,)
        num_steps = 6
        neuron = LIF(shape=shape,
                     vth=3,
                     bias_mant=1)

        rcnd = RunSteps(num_steps=num_steps)
        rcfg = Loihi1SimCfg(select_tag="floating_pt")

        # Probe voltage of LIF with the first monitor
        monitor1.probe(target=neuron.v, num_steps=num_steps)

        # Probe spike output of LIF with the second monitor
        monitor2.probe(target=neuron.s_out, num_steps=num_steps)

        # Run all connected processes
        neuron.run(condition=rcnd, run_cfg=rcfg)

        # Get data from both monitor
        data1 = monitor1.get_data()
        data2 = monitor2.get_data()

        neuron.stop()

        # Access the relevant data in the corresponding data dicts
        volt_data = data1[neuron.name][neuron.v.name]
        spike_data = data2[neuron.name][neuron.s_out.name]

        # Check if this data match the expected data
        self.assertTrue(np.all(volt_data == np.array([[1, 2, 3, 0, 1, 2]]).T))
        self.assertTrue(np.all(spike_data == np.array([[0, 0, 0, 1, 0, 0]]).T))

    def test_monitor_collects_voltage_and_spike_data_from_population_lif(self):
        """Check if two different Monitor process can monitor voltage (Var) and
        s_out (OutPort) of a population of two LIF neurons. Check the
        collected data with expected data.
        Note: The LIF neurons integrate the given bias, voltage accumulates
        and once pass the threshold, there will be a spike outputted and
        voltage will be reset to zero.
        """

        # Setup two Monitor Processes and LIF Process (population of neurons)
        # that will be monitored
        monitor1 = Monitor()
        monitor2 = Monitor()
        shape = (2,)
        num_steps = 6
        neuron = LIF(shape=shape,
                     vth=3,
                     bias_mant=1)

        # Probe voltage of LIF neurons with the first monitor
        monitor1.probe(target=neuron.v, num_steps=num_steps)

        # Probe spike output of LIF neurons with the second monitor
        monitor2.probe(target=neuron.s_out, num_steps=num_steps)

        # Run all connected processes
        neuron.run(condition=RunSteps(num_steps=num_steps),
                   run_cfg=Loihi1SimCfg(select_tag="floating_pt"))

        # Get data from both monitor
        data1 = monitor1.get_data()
        data2 = monitor2.get_data()

        neuron.stop()

        # Access the relevant data in the corresponding data dicts
        volt_data = data1[neuron.name][neuron.v.name]
        spike_data = data2[neuron.name][neuron.s_out.name]

        # Check if this data match the expected data
        self.assertTrue(np.all(volt_data == np.array([[1, 2, 3, 0, 1, 2],
                                                      [1, 2, 3, 0, 1, 2]]).T))
        self.assertTrue(np.all(spike_data == np.array([[0, 0, 0, 1, 0, 0],
                                                       [0, 0, 0, 1, 0, 0]]).T))

    def test_monitor_collects_data_from_2D_population_lif(self):
        """Check if two different Monitor process can monitor voltage (Var) and
        s_out (OutPort) of a 2-D population of 4 LIF neurons. Check the
        collected data with expected data.
        Note: The LIF neurons integrate the given bias, voltage accumulates
        and once pass the threshold, there will be a spike outputted and
        voltage will be reset to zero.
        """

        # Setup two Monitor Processes and LIF Process (population of neurons)
        # that will be monitored
        monitor1 = Monitor()
        monitor2 = Monitor()
        shape = (2, 2)
        num_steps = 6
        neuron = LIF(shape=shape,
                     vth=3,
                     bias_mant=1)

        # Probe voltage of LIF neurons with the first monitor
        monitor1.probe(target=neuron.v, num_steps=num_steps)

        # Probe spike output of LIF neurons with the second monitor
        monitor2.probe(target=neuron.s_out, num_steps=num_steps)

        # Run all connected processes
        neuron.run(condition=RunSteps(num_steps=num_steps),
                   run_cfg=Loihi1SimCfg(select_tag="floating_pt"))

        # Get data from both monitor
        data1 = monitor1.get_data()
        data2 = monitor2.get_data()

        neuron.stop()

        # Access the relevant data in the corresponding data dicts
        volt_data = data1[neuron.name][neuron.v.name]
        spike_data = data2[neuron.name][neuron.s_out.name]

        # Check if this data match the expected data
        expected_voltages = np.tile(
            np.expand_dims(np.array([1, 2, 3, 0, 1, 2]), (1, 2)), (1,) + shape)
        expected_spikes = np.tile(
            np.expand_dims(np.array([0, 0, 0, 1, 0, 0]), (1, 2)), (1,) + shape)

        self.assertTrue(np.all(volt_data == expected_voltages))
        self.assertTrue(np.all(spike_data == expected_spikes))

    def test_proc_params_accessible_in_proc_model(self):
        """Check if proc_params are accessible in ProcessModel. This
        functionality is necessary to access dynamically created Ports/Vars"""

        monitor = Monitor()

        # Set some dummy proc_params to be transferred to ProcessModel
        monitor.proc_params = {"test": 0}

        # Compile
        c = Compiler()
        exe = c.compile(monitor, Loihi1SimCfg())

        # Check if built model has these proc_params
        self.assertEqual(next(iter(exe.proc_builders)).proc_params,
                         monitor.proc_params)


if __name__ == '__main__':
    unittest.main()
