# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest

from lava.magma.core.decorator import implements, requires
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


# A minimal process with a Var and a RefPort, VarPort
class P1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref1 = RefPort(shape=(3,))
        self.var1 = Var(shape=(2,), init=17)
        self.var_port_var1 = VarPort(self.var1)


# A minimal process with 2 Vars and a RefPort, VarPort
class P2(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.var2 = Var(shape=(3,), init=4)
        self.var_port_var2 = VarPort(self.var2)
        self.ref2 = RefPort(shape=(2,))
        self.var3 = Var(shape=(2,), init=1)


# A minimal PyProcModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
class PyProcModel1(PyLoihiProcessModel):
    ref1: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)
    var1: np.ndarray = LavaPyType(np.ndarray, np.int32)
    var_port_var1: PyVarPort = LavaPyType(PyVarPort.VEC_DENSE, int)

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        if self.time_step > 1:
            ref_data = np.array([5, 5, 5]) + self.time_step
            self.ref1.write(ref_data)


# A minimal PyProcModel implementing P2
@implements(proc=P2, protocol=LoihiProtocol)
@requires(CPU)
class PyProcModel2(PyLoihiProcessModel):
    ref2: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)
    var2: np.ndarray = LavaPyType(np.ndarray, np.int32)
    var_port_var2: PyVarPort = LavaPyType(PyVarPort.VEC_DENSE, int)
    var3: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        if self.time_step > 1:
            self.var3 = self.ref2.read()


# A simple RunConfig selecting always the first found process model
class MyRunCfg(RunConfig):
    def select(self, proc, proc_models):
        return proc_models[0]


class TestRefVarPorts(unittest.TestCase):
    def test_unconnected_Ref_Var_ports(self):
        """RefPorts and VarPorts defined in ProcessModels, but not connected
        should not lead to an error."""
        sender = P1()

        # No connections are made

        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [sender])

        # The process should compile and run without error (not doing anything)
        sender.run(RunSteps(num_steps=3, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        sender.stop()

    def test_explicit_Ref_Var_port_write(self):
        """Tests the connection of a RefPort to an explicitly created VarPort.
        The RefPort sends data after the first time step to the VarPort,
        starting with (5 + current time step) = 7). The initial value of the
        var is 4. We read out the value after each time step."""

        sender = P1()
        recv = P2()

        # Connect RefPort with explicit VarPort
        sender.ref1.connect(recv.var_port_var2)

        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [sender, recv])

        # First time step, no data is sent
        sender.run(RunSteps(num_steps=1, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        # Initial value is expected
        self.assertTrue(np.all(recv.var2.get() == np.array([4., 4., 4.])))
        # Second time step, data is sent (7)
        sender.run(RunSteps(num_steps=1, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        self.assertTrue(np.all(recv.var2.get() == np.array([7., 7., 7.])))
        # Third time step, data is sent (8)
        sender.run(RunSteps(num_steps=1, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        self.assertTrue(np.all(recv.var2.get() == np.array([8., 8., 8.])))
        # Fourth time step, data is sent (9)
        sender.run(RunSteps(num_steps=1, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        self.assertTrue(np.all(recv.var2.get() == np.array([9., 9., 9.])))
        sender.stop()

    def test_implicit_Ref_Var_port_write(self):
        """Tests the connection of a RefPort to an implicitly created VarPort.
        The RefPort sends data after the first time step to the VarPort,
        starting with (5 + current time step) = 7). The initial value of the
        var is 4. We read out the value after each time step."""

        sender = P1()
        recv = P2()

        # Connect RefPort with Var using an implicit VarPort
        sender.ref1.connect_var(recv.var2)

        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [sender, recv])

        # First time step, no data is sent
        sender.run(RunSteps(num_steps=1, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        # Initial value is expected
        self.assertTrue(np.all(recv.var2.get() == np.array([4., 4., 4.])))
        # Second time step, data is sent (7)
        sender.run(RunSteps(num_steps=1, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        self.assertTrue(np.all(recv.var2.get() == np.array([7., 7., 7.])))
        # Third time step, data is sent (8)
        sender.run(RunSteps(num_steps=1, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        self.assertTrue(np.all(recv.var2.get() == np.array([8., 8., 8.])))
        # Fourth time step, data is sent (9)
        sender.run(RunSteps(num_steps=1, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        self.assertTrue(np.all(recv.var2.get() == np.array([9., 9., 9.])))
        sender.stop()

    def test_explicit_Ref_Var_port_read(self):
        """Tests the connection of a RefPort to an explicitly created VarPort.
        The RefPort "ref_read" reads data after the first time step of the
        VarPort "var_port_read" which has the value of the Var "v" (= 17) and
        writes this value into the Var "var_read". The initial value of the var
        "var_read" is 1. At time step 2 the value of "var_read" is 17."""

        sender = P1()
        recv = P2()

        # Connect RefPort with explicit VarPort
        recv.ref2.connect(sender.var_port_var1)

        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [sender, recv])

        # First time step, no read
        sender.run(RunSteps(num_steps=1, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        # Initial value (1) is expected
        self.assertTrue(np.all(recv.var3.get() == np.array([1., 1.])))
        # Second time step, the RefPort read from the VarPort and wrote the
        # Result in "var_read" (= 17)
        sender.run(RunSteps(num_steps=1, blocking=True),
                   MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        self.assertTrue(
            np.all(recv.var3.get() == np.array([17., 17.])))
        sender.stop()

    def test_implicit_Ref_Var_port_read(self):
        """Tests the connection of a RefPort to an implicitly created VarPort.
        The RefPort "ref_read" reads data after the first time step of the
        of the Var "v" (= 17) using an implicit VarPort and writes this value
        into the Var "var_read". The initial value of the var "var_read" is 1.
        At time step 2 the value of "var_read" is 17."""

        sender = P1()
        recv = P2()

        # Connect RefPort with explicit VarPort
        recv.ref2.connect_var(sender.var1)

        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [sender, recv])

        # First time step, no read
        recv.run(RunSteps(num_steps=1, blocking=True),
                 MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        # Initial value (1) is expected
        self.assertTrue(np.all(recv.var3.get() == np.array([1., 1.])))
        # Second time step, the RefPort read from the VarPort and wrote the
        # Result in "var_read" (= 17)
        recv.run(RunSteps(num_steps=1, blocking=True),
                 MyRunCfg(custom_sync_domains=[simple_sync_domain]))
        self.assertTrue(
            np.all(recv.var3.get() == np.array([17., 17.])))
        recv.stop()


if __name__ == '__main__':
    unittest.main()
