# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import unittest

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_conditions import RunSteps


# A minimal process with an OutPort
class P1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out = OutPort(shape=(2,))


# A minimal process with a Var and an InPort
class P2(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inp = InPort(shape=(2,))


# A minimal PyProcModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModel1(PyLoihiProcessModel):
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run_spk(self):
        # Raise exception
        raise AssertionError("P1")
        # raise AssertionError("{}: All the information".format(self.__class__))


# A minimal PyProcModel implementing P2
@implements(proc=P2, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModel2(PyLoihiProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    def run_spk(self):
        # Raise exception
        raise TypeError("P2")
        # raise TypeError("{}: All the information".format(self.__class__))


class TestExceptionHandling(unittest.TestCase):
    def test_one_pm(self):
        """Checks the forwarding of exceptions within a ProcessModel to the
        runtime."""

        # Create an instance of P1
        proc = P1()

        # Run the network for 2 time steps
        proc.run(condition=RunSteps(num_steps=2), run_cfg=Loihi1SimCfg())

        # Except exception raised

        proc.stop()

    def test_two_pm(self):
        """Checks the forwarding of exceptions within two ProcessModel to the
        runtime."""

        # Create a sender instance of P1 and a receiver instance of P2
        sender = P1()
        recv = P2()

        # Connect sender with receiver
        sender.out.connect(recv.inp)

        # Run the network for 2 time steps
        sender.run(condition=RunSteps(num_steps=2), run_cfg=Loihi1SimCfg())

        # The expected value of var in the recv is [1, 2]

        sender.stop()


if __name__ == '__main__':
    unittest.main()