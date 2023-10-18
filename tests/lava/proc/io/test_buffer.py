# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.process.variable import Var
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.injector import Injector
from lava.proc.io.extractor import Extractor
from lava.proc.lif.process import LIF
from lava.proc.io.buffer import Buffer


class TestBuffer(unittest.TestCase):
    def test_create_buffer(self):
        buffer = Buffer()
        self.assertEqual(buffer.length, 100)
        self.assertEqual(buffer.overflow, "raise_error")

    def test_add_var(self):
        buffer = Buffer(length=50)
        shape = (10,)
        test_shape = (10, 50)
        buffer.add_var(name="test", shape=shape)
        self.assertIn(buffer.test, buffer.vars)
        self.assertEqual(buffer.test.shape, test_shape)

    def test_connect(self):
        buffer = Buffer(length=10)
        var_shape = (3,)
        buffer_shape = (3, 10)
        test_var = Var(shape=var_shape)
        self.assertFalse(hasattr(buffer, "test"))
        self.assertFalse(hasattr(buffer, "testRefPort0"))
        buffer.connect("test", test_var)
        self.assertTrue(hasattr(buffer, "test"))
        self.assertTrue(hasattr(buffer, "testRefPort0"))
        self.assertIn(buffer.test, buffer.vars)
        self.assertIn(buffer.testRefPort0, buffer.ref_ports)
        self.assertEqual(buffer.test.shape, buffer_shape)
        self.assertEqual(buffer.testRefPort0.shape, var_shape)

    def test_connect_from(self):
        buffer = Buffer(length=10)
        var_shape = (3,)
        buffer_shape = (3, 10)
        test_out_port = OutPort(shape=var_shape)
        buffer.connect_from("v", test_out_port)
        self.assertTrue(hasattr(buffer, "v"))
        self.assertEqual(buffer.v.shape, buffer_shape)
        self.assertTrue(buffer.vInPort0 in buffer.in_ports)

    def test_run_single_input_buffer(self):
        """Test input Buffer."""
        num_steps = 10
        injector = Injector(shape=(1,))
        buffer = Buffer(length=10)
        buffer.connect_from("v0", injector.out_port)
        buffer.create_runtime(run_cfg=Loihi2SimCfg())
        buffer.run(condition=RunSteps(num_steps, blocking=False))
        for t in range(num_steps):
            injector.send(np.full((1,), t))
        buffer.wait()
        data = buffer.v0.get().flatten().astype(int).tolist()
        buffer.stop()
        self.assertSequenceEqual(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_run_single_input_buffer_2vars(self):
        """Test to ensure that a Buffer can connect to multiple Vars."""
        num_steps = 10
        injector = Injector(shape=(1,))
        lif = LIF(shape=(1,), du=1)
        buffer = Buffer(length=10)
        injector.out_port.connect(lif.a_in)
        buffer.connect_from("u", lif.u)
        buffer.connect_from("v", lif.v)
        buffer.create_runtime(run_cfg=Loihi2SimCfg())
        buffer.run(condition=RunSteps(num_steps, blocking=False))
        for t in range(num_steps):
            injector.send(np.full((1,), t))
        buffer.wait()
        udata = buffer.u.get().flatten().astype(int).tolist()
        vdata = buffer.v.get().flatten().astype(int).tolist()
        buffer.stop()
        self.assertSequenceEqual(udata, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertSequenceEqual(vdata, [0, 1, 3, 6, 10, 0, 6, 0, 8, 0])

    def test_multiple_buffers(self):
        """Test to ensure that two Buffers in the same process graph
        do not interfere with one another due to dynamic Vars/Ports."""
        num_steps = 10
        injector = Injector(shape=(1,))
        lif = LIF(shape=(1,), du=1)
        ubuffer = Buffer(length=10)
        vbuffer = Buffer(length=10)
        injector.out_port.connect(lif.a_in)
        ubuffer.connect_from("u", lif.u)
        vbuffer.connect_from("v", lif.v)
        ubuffer.create_runtime(run_cfg=Loihi2SimCfg())
        ubuffer.run(condition=RunSteps(num_steps, blocking=False))
        for t in range(num_steps):
            injector.send(np.full((1,), t))
        ubuffer.wait()
        udata = ubuffer.u.get().flatten().astype(int).tolist()
        vdata = vbuffer.v.get().flatten().astype(int).tolist()
        ubuffer.stop()
        self.assertSequenceEqual(udata, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertSequenceEqual(vdata, [0, 1, 3, 6, 10, 0, 6, 0, 8, 0])

    def test_run_single_output_buffer(self):
        """Test output Buffer."""
        num_steps = 10
        extractor = Extractor(shape=(1,))
        buffer = Buffer(length=10)
        vdata = np.array(range(10)).reshape((1, 10))
        buffer.connect("v", extractor.in_port, init=vdata)
        buffer.create_runtime(run_cfg=Loihi2SimCfg())
        buffer.run(condition=RunSteps(num_steps, blocking=False))
        for t in range(num_steps):
            self.assertEqual(extractor.receive(), vdata[0, t])
        buffer.wait()
        buffer.stop()


if __name__ == "__main__":
    unittest.main()
