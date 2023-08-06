# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import List
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
        self.assertEqual(buffer.overflow, 'raise_error')

    def test_connect_buffer(self):
        buffer = Buffer()
        # Test connecting to a Var
        var_shape = (10,)
        buffer_shape = (10, 100)
        test_var = Var(shape=var_shape)
        self.assertFalse(hasattr(buffer, 'Var1000'))
        self.assertFalse(hasattr(buffer, 'Ref1000'))
        buffer.connect(test_var)
        self.assertTrue(hasattr(buffer, 'Var1000'))
        self.assertTrue(hasattr(buffer, 'Ref1000'))
        self.assertIn(buffer.Var1000, buffer.vars)
        self.assertIn(buffer.Ref1000, buffer.ref_ports)
        self.assertEqual(buffer.Ref1000.shape, var_shape)
        self.assertEqual(buffer.Var1000.shape, buffer_shape)
        # Test connecting to an OutPort
        test_out_port = OutPort(shape=var_shape)
        v = buffer.connect(test_out_port)
        self.assertEqual(v.shape, buffer_shape)
        self.assertTrue(hasattr(buffer, v.name))
        self.assertEqual(len(list(buffer.in_ports)), 1)

    def test_run_buffer(self):
        num_steps = 10
        injector = Injector(shape=(1,))
        buffer = Buffer(length=10)
        v0 = buffer.connect(injector.out_port)
        buffer.create_runtime(run_cfg=Loihi2SimCfg())
        buffer.run(condition=RunSteps(num_steps, blocking=False))
        for t in range(num_steps):
            injector.send(np.full((1,), t))
        buffer.wait()
        data = v0.get().flatten().astype(int).tolist()
        buffer.stop()
        self.assertSequenceEqual(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_buffer_with_2vars(self):
        """Test to ensure that a Buffer can connect to multiple Vars."""
        num_steps = 10
        injector = Injector(shape=(1,))
        lif = LIF(shape=(1,), du=1)
        buffer = Buffer(length=10)
        injector.out_port.connect(lif.a_in)
        u = buffer.connect(lif.u)
        v = buffer.connect(lif.v)
        buffer.create_runtime(run_cfg=Loihi2SimCfg())
        buffer.run(condition=RunSteps(num_steps, blocking=False))
        for t in range(num_steps):
            injector.send(np.full((1,), t))
        buffer.wait()
        udata = u.get().flatten().astype(int).tolist()
        vdata = v.get().flatten().astype(int).tolist()
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
        u = ubuffer.connect(lif.u)
        v = vbuffer.connect(lif.v)
        ubuffer.create_runtime(run_cfg=Loihi2SimCfg())
        ubuffer.run(condition=RunSteps(num_steps, blocking=False))
        for t in range(num_steps):
            injector.send(np.full((1,), t))
        ubuffer.wait()
        udata = u.get().flatten().astype(int).tolist()
        vdata = v.get().flatten().astype(int).tolist()
        ubuffer.stop()
        self.assertSequenceEqual(udata, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertSequenceEqual(vdata, [0, 1, 3, 6, 10, 0, 6, 0, 8, 0])

    def test_output_buffer(self):
        num_steps = 10
        extractor = Extractor(shape=(1,))
        buffer = Buffer(length=10)
        vdata = np.array(range(10)).reshape((1, 10))
        v = buffer.connect(extractor.in_port, init=vdata)
        buffer.create_runtime(run_cfg=Loihi2SimCfg())
        buffer.run(condition=RunSteps(num_steps, blocking=False))
        for t in range(num_steps):
            self.assertEqual(extractor.receive(), vdata[0,t])
        buffer.wait()
        buffer.stop()


if __name__ == '__main__':
    unittest.main()