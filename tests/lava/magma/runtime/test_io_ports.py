# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_conditions import RunSteps


# A minimal hierarchical process with an OutPort
class HP1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.h_var = Var(shape=(2,))
        self.h_out = OutPort(shape=(2,))


# A minimal hierarchical process with an InPort
class HP2(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.h_var = Var(shape=(2, ))
        self.h_inp = InPort(shape=(2,))


# A minimal process with an OutPort
class P1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.var = Var(shape=(2,), init=4)
        self.out = OutPort(shape=(2,))


# A minimal process with a Var and an InPort
class P2(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.var = Var(shape=(2,), init=5)
        self.inp = InPort(shape=(2,))


# A minimal hierarchical PyProcModel implementing HP2
@implements(proc=HP1)
class PyProcModelHP1(AbstractSubProcessModel):

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""

        # Connect the OutPort of the hierarchical process with the OutPort of
        # the nested process
        self.p1 = P1()
        self.p1.out.connect(proc.out_ports.h_out)
        # Reference h_var with var of the nested process
        proc.vars.h_var.alias(self.p1.var)


# A minimal hierarchical PyProcModel implementing HP2
@implements(proc=HP2)
class PyProcModelHP2(AbstractSubProcessModel):

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""

        self.p2 = P2()
        # Connect the InPort of the hierarchical process with the InPort of
        # the nested process
        proc.in_ports.h_inp.connect(self.p2.inp)
        # Reference h_var with var of the nested process
        proc.vars.h_var.alias(self.p2.var)


# A minimal PyProcModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModel1(PyLoihiProcessModel):
    var: np.ndarray = LavaPyType(np.ndarray, np.int32)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run_spk(self):
        # Send data
        data = np.array([1, 2])
        self.var = data + 2
        self.out.send(data)


# A minimal PyProcModel implementing P2
@implements(proc=P2, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModel2(PyLoihiProcessModel):
    var: np.ndarray = LavaPyType(np.ndarray, np.int32)
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    def run_spk(self):
        # Receive data
        data = self.inp.recv()
        # Store data in var
        self.var = data


class RecursiveProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_port = OutPort(shape=(1,))
        self.in_port = InPort(shape=(1,))


@implements(proc=RecursiveProcess, protocol=LoihiProtocol)
@requires(CPU)
class RecursiveProcessModel(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run_spk(self):
        # !!!! Receiving First Before Sending Will Cause Hung Behaviour !!!!
        # Receive data
        data = self.in_port.recv()
        # Send data
        self.out_port.send(data)

    # def run_spk(self):
    #     # This is valid implementation as we do send some data to unblock
    #     data = np.ones(shape=(1,))
    #     # Send data
    #     self.out_port.send(data)
    #     # Receive data
    #     data = self.in_port.recv()


class TestIOPorts(unittest.TestCase):
    def test_send_recv(self):
        """Checks if sending data via an OutPort in P1 to an InPort in P2
        works. The data is received in P2 and stored in its Var var to compare
        it to its expected value."""

        # Create a sender instance of P1 and a receiver instance of P2
        sender = P1()
        recv = P2()

        # Connect sender with receiver
        sender.out.connect(recv.inp)

        # Run the network for 2 time steps
        sender.run(condition=RunSteps(num_steps=2),
                   run_cfg=Loihi1SimCfg())

        # The expected value of var in the recv is [1, 2]
        self.assertTrue(np.all(recv.var.get() == np.array([1, 2])))
        sender.stop()

    def test_merging(self):
        """Checks sending data via three OutPort of P1 instances to an InPort
        of a P2 instance. The data is received in P2 and stored in its Var var
        to compare it to its expected value. Multiple inputs should get added
        up."""

        # Create 3 sender instance of P1 and a receiver instance of P2
        sender1 = P1()
        sender2 = P1()
        sender3 = P1()
        recv = P2()

        # Connect all senders with receiver
        recv.inp.connect_from([sender1.out, sender2.out, sender3.out])

        # Run the network for 2 time steps
        recv.run(condition=RunSteps(num_steps=2), run_cfg=Loihi1SimCfg())

        # The expected value of var in the recv is [3, 6]
        self.assertTrue(np.all(recv.var.get() == np.array([3, 6])))
        recv.stop()

    def test_branching(self):
        """Checks sending data via an OutPort of a P1 instance to InPorts
        of three P2 instances. The data is received in each P2 instance and
        stored in its respected Var var to compare it to its expected value.
        The same output should be received for each receiver."""

        # Create a sender instance of P1 and three receiver instances of P2
        sender = P1()
        recv1 = P2()
        recv2 = P2()
        recv3 = P2()

        # Connect the sender with all receivers
        sender.out.connect([recv1.inp, recv2.inp, recv3.inp])

        # Run the network for 2 time steps
        sender.run(condition=RunSteps(num_steps=2), run_cfg=Loihi1SimCfg())

        # The expected value of var in the recv is [1, 2] for all receivers
        self.assertTrue(np.all(recv1.var.get() == np.array([1, 2])))
        self.assertTrue(np.all(recv2.var.get() == np.array([1, 2])))
        self.assertTrue(np.all(recv3.var.get() == np.array([1, 2])))
        sender.stop()

    def test_send_recv_hierarchical(self):
        """Checks if sending data via an OutPort of the hierarchical process HP1
        to an InPort of the hierarchical process HP2 works. The OutPort of HP1
        receives its data from the OutPort of its nested process P1. The data is
        received in HP2 and forwarded to the InPort of its nested process P2.
        The data is then stored in Var var of P2. HP2 has a Var h_var which
        aliases the Var var in order to compare the data to its expected value.
        """

        # Create a sender instance of HP1 and a receiver instance of HP2
        sender = HP1()
        recv = HP2()

        # Connect sender with receiver
        sender.h_out.connect(recv.h_inp)

        # Run the network for 2 time steps
        sender.run(condition=RunSteps(num_steps=2),
                   run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        # The expected value of var in the recv is [1, 2]
        self.assertTrue(np.all(recv.h_var.get() == np.array([1, 2])))
        sender.stop()

    def test_merging_hierarchical(self):
        """Checks if sending data via an OutPort of three instances of the
        hierarchical process HP1 to an InPort of an instance of hierarchical
        process HP2 works. The OutPort of HP1 receives its data from the OutPort
        of its nested process P1. The data is received in HP2 and forwarded to
        the InPort of its nested process P2. The data is then stored in Var var
        of P2. HP2 has a Var h_var which aliases the Var var in order to compare
        the data to its expected value. Multiple inputs should get added up.
        """

        # Create 3 sender instance of HP1 and a receiver instance of HP2
        sender1 = HP1()
        sender2 = HP1()
        sender3 = HP1()

        recv = HP2()

        # Connect all senders with receiver
        recv.h_inp.connect_from([sender1.h_out, sender2.h_out, sender3.h_out])

        # Run the network for 2 time steps
        recv.run(condition=RunSteps(num_steps=2),
                 run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        # The expected value of var in the recv is [3, 6]
        self.assertTrue(np.all(recv.h_var.get() == np.array([3, 6])))
        recv.stop()

    def test_branching_hierarchical(self):
        """Checks if sending data via an OutPort of an instance of the
        hierarchical process HP1 to an InPort of three instances of hierarchical
        process HP2 works. The OutPort of HP1 receives its data from the OutPort
        of its nested process P1. The data is received in HP2 and forwarded to
        the InPort of its nested process P2. The data is then stored in Var var
        of P2. HP2 has a Var h_var which aliases the Var var in order to compare
        the data to its expected value. The same output should be received for
        each receiver."""

        # Create a sender instance of HP1 and 3 receiver instances of HP2
        sender = HP1()
        recv1 = HP2()
        recv2 = HP2()
        recv3 = HP2()

        # Connect all senders with receiver
        sender.h_out.connect([recv1.h_inp, recv2.h_inp, recv3.h_inp])

        # Run the network for 2 time steps
        sender.run(condition=RunSteps(num_steps=2),
                   run_cfg=Loihi1SimCfg(select_sub_proc_model=True))

        # The expected value of var in the recv is [1, 2] for all receivers
        self.assertTrue(np.all(recv1.h_var.get() == np.array([1, 2])))
        self.assertTrue(np.all(recv2.h_var.get() == np.array([1, 2])))
        self.assertTrue(np.all(recv3.h_var.get() == np.array([1, 2])))
        sender.stop()

    def test_dangling_input(self):
        """Checks if the hierarchical process works with dangling input i.e.
        input not connected at all."""
        receiver = HP2()
        receiver.run(condition=RunSteps(num_steps=2),
                     run_cfg=Loihi1SimCfg(select_sub_proc_model=True))
        self.assertTrue(np.all(receiver.h_var.get() == np.array([0, 0])))
        receiver.stop()

    def test_dangling_output(self):
        """Checks if the hierarchical process works with dangling output i.e.
        output not connected at all."""
        sender = HP1()
        sender.run(condition=RunSteps(num_steps=2),
                   run_cfg=Loihi1SimCfg(select_sub_proc_model=True))
        self.assertTrue(np.all(sender.h_var.get() == np.array([3, 4])))
        sender.stop()

    @unittest.skip("Only for Testing Blocked Receivers")
    def test_recursive_blocking(self):
        sender = RecursiveProcess(log_config=LogConfig(level=20))
        receiver = RecursiveProcess()

        sender.out_port.connect(receiver.in_port)
        receiver.out_port.connect(sender.in_port)

        # Output Long Timeouts within 10s and Short Timeouts within 2s
        # Defaults are higher
        sender.run(condition=RunSteps(2),
                   run_cfg=Loihi1SimCfg(),
                   compile_config={"long_event_timeout": 2.0,
                                   "short_event_timeout": 1.0})
        sender.stop()


if __name__ == '__main__':
    unittest.main()
