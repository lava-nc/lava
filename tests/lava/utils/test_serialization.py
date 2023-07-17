# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
import tempfile
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.utils.serialization import save, load
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.decorator import implements
from lava.magma.core.process.variable import Var


# A minimal hierarchical process
class HP(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lif_in_v = Var(shape=(2,))
        self.lif_out_u = Var(shape=(3,))


# A minimal hierarchical PyProcModel implementing HP
@implements(proc=HP)
class PyProcModelHP(AbstractSubProcessModel):

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""

        pre_size = 2
        post_size = 3

        weights = np.ones((post_size, pre_size))

        self.lif_in = LIF(shape=(pre_size,), bias_mant=100, vth=120,
                          name="LIF_neuron input")
        self.dense = Dense(weights=weights * 10, name="Dense")
        self.lif_out = LIF(shape=(post_size,), bias_mant=0, vth=50000,
                           name="LIF_neuron output")

        self.lif_in.s_out.connect(self.dense.s_in)
        self.dense.a_out.connect(self.lif_out.a_in)

        proc.vars.lif_in_v.alias(self.lif_in.v)
        proc.vars.lif_out_u.alias(self.lif_out.u)


class TestSerialization(unittest.TestCase):
    def test_save_input_validation(self):
        """Checks the input validation of save()."""

        # Parameter processes needs to be AbstractProcess or list of
        # AbstractProcess
        with self.assertRaises(TypeError):
            save(processes=None, filename="test")

        # Parameter filename needs to be string
        with self.assertRaises(TypeError):
            save(processes=[], filename=1)

        # Parameter executable needs to be Executable
        with self.assertRaises(TypeError):
            save(processes=[], filename="test", executable=1)

    def test_load_input_validation(self):
        """Checks the input validation of load()."""

        # Parameter filename needs to be string
        with self.assertRaises(TypeError):
            load(filename=1)

    def test_save_load_processes(self):
        """Checks storing and loading processes."""

        weights = np.ones((2, 3))

        # Create some processes
        dense = Dense(weights=weights, name="Dense")
        lif_procs = []
        for i in range(5):
            lif_procs.append(LIF(shape=(1,), name="LIF" + str(i)))

        # Store the processes in file test.pickle
        with tempfile.TemporaryDirectory() as tmpdirname:
            save(lif_procs + [dense], tmpdirname + "test")
            dense = None

            # Load the processes again from test.pickle
            procs, _ = load(tmpdirname + "test.pickle")

        dense = procs[-1]

        # Check if the processes have the same parameters
        self.assertTrue(np.all(dense.weights.get() == weights))
        self.assertTrue(dense.name == "Dense")

        for i in range(5):
            self.assertTrue(isinstance(procs[i], LIF))
            self.assertTrue(procs[i].name == "LIF" + str(i))

    def test_save_load_executable(self):
        """Checks storing and loading of executable."""

        # Create a process
        lif = LIF(shape=(1,), name="ExecLIF")

        # Create an executable
        ex = lif.compile(run_cfg=Loihi2SimCfg())

        # Store the executable in file test.pickle
        with tempfile.TemporaryDirectory() as tmpdirname:
            save([], tmpdirname + "test", executable=ex)

            # Load the executable from test.pickle
            p, executable = load(tmpdirname + "test.pickle")

        # Check if the executable reflects the inital process
        self.assertTrue(p == [])
        loaded_lif = executable.process_list[0]
        self.assertTrue(lif.name == loaded_lif.name)

    def test_save_load_hierarchical_proc(self):
        """Checks saving, loading and execution of a workload using a
        hierarchical process."""

        num_steps = 5
        output_lif_in_v = np.zeros(shape=(2, num_steps))
        output_lif_out_u = np.zeros(shape=(3, num_steps))

        # Create hierarchical process
        proc = HP()

        # Create executable
        ex = proc.compile(run_cfg=Loihi2SimCfg())

        # Store executable and run it
        with tempfile.TemporaryDirectory() as tmpdirname:
            save(proc, tmpdirname + "test", ex)

            proc.create_runtime(executable=ex)
            try:
                for i in range(num_steps):
                    proc.run(condition=RunSteps(num_steps=1))

                    output_lif_in_v[:, i] = proc.lif_in_v.get()
                    output_lif_out_u[:, i] = proc.lif_out_u.get()
            finally:
                proc.stop()

            # Load executable again
            proc_loaded, ex_loaded = load(tmpdirname + "test.pickle")

        output_lif_in_v_loaded = np.zeros(shape=(2, num_steps))
        output_lif_out_u_loaded = np.zeros(shape=(3, num_steps))

        # Run the loaded executable
        proc_loaded.create_runtime(executable=ex_loaded)
        try:
            for i in range(num_steps):
                proc_loaded.run(condition=RunSteps(num_steps=1))

                output_lif_in_v_loaded[:, i] = proc_loaded.lif_in_v.get()
                output_lif_out_u_loaded[:, i] = proc_loaded.lif_out_u.get()
        finally:
            proc_loaded.stop()

        # Compare results from inital run and run of loaded executable
        self.assertTrue(np.all(output_lif_in_v == output_lif_in_v_loaded))
        self.assertTrue(np.all(output_lif_out_u == output_lif_out_u_loaded))


if __name__ == '__main__':
    unittest.main()
