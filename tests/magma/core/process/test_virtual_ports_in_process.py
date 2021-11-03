# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
import unittest
import numpy as np

from lava.magma.core.decorator import has_models, requires
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import (
    InPort,
    OutPort,
)
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import RunConfig


class TestVirtualPorts(unittest.TestCase):
    """Contains unit tests around virtual ports created as part of a Process."""

    def setUp(self) -> None:
        # minimal process with an OutPort
        pass

    @unittest.skip("skip while solved")
    def test_multi_inports(self):
        sender = P1()
        recv1 = P2()
        recv2 = P2()
        recv3 = P2()

        # An OutPort can connect to multiple InPorts
        # Either at once...
        sender.out.connect([recv1.inp, recv2.inp, recv3.inp])

        sender = P1()
        recv1 = P2()
        recv2 = P2()
        recv3 = P2()

        # ... or consecutively
        sender.out.connect(recv1.inp)
        sender.out.connect(recv2.inp)
        sender.out.connect(recv3.inp)
        sender.run(RunSteps(num_steps=2), MyRunCfg())

    @unittest.skip("skip while solved")
    def test_reshape(self):
        """Checks reshaping of a port."""
        sender = P1(shape=(1, 6))
        recv = P2(shape=(2, 3))

        # Using reshape(..), ports with different shape can be connected as
        # long as total number of elements does not change
        sender.out.reshape((2, 3)).connect(recv.inp)
        sender.run(RunSteps(num_steps=2), MyRunCfg())

    @unittest.skip("skip while solved")
    def test_concat(self):
        """Checks concatenation of ports."""
        sender1 = P1(shape=(1, 2))
        sender2 = P1(shape=(1, 2))
        sender3 = P1(shape=(1, 2))
        recv = P2(shape=(3, 2))

        # concat_with(..) concatenates calling port (sender1.out) with
        # other ports (sender2.out, sender3.out) along given axis
        cp = sender1.out.concat_with([sender2.out, sender3.out], axis=0)

        # The return value is a virtual ConcatPort which can be connected
        # to the input port
        cp.connect(recv.inp)
        sender1.run(RunSteps(num_steps=2), MyRunCfg())


# A minimal PyProcModel implementing P1
@requires(CPU)
class PyProcModelA(AbstractPyProcessModel):
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def run(self):
        data = np.asarray([1, 2, 3])
        self.out.send(data)
        print("Sent output data of P1: ", str(data))


# A minimal PyProcModel implementing P2
@requires(CPU)
class PyProcModelB(AbstractPyProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    def run(self):
        in_data = self.inp.recv()
        print("Received input data for P2: ", str(in_data))


# minimal process with an OutPort
@has_models(PyProcModelA)
class P1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (3,))
        self.out = OutPort(shape=shape)


# minimal process with an InPort
@has_models(PyProcModelB)
class P2(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get('shape', (3,))
        self.inp = InPort(shape=shape)


class MyRunCfg(RunConfig):
    def select(self, proc, proc_models):
        return proc_models[0]


if __name__ == '__main__':
    unittest.main()
