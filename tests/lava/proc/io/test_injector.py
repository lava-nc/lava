import numpy as np
import unittest
import threading
import time
from time import sleep
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous

from lava.proc.io.in_bridge import AsyncInjector, SyncInjector


class Recv(AbstractProcess):
    """Process that receives arbitrary dense data and stores it in a Var.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort and Var.
    """

    def __init__(self, shape: tuple[...]):
        super().__init__(shape=shape)

        self.var = Var(shape=shape, init=0)
        self.in_port = InPort(shape=shape)


@implements(proc=Recv, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLoihiFloatingPointRecvProcessModel(PyLoihiProcessModel):
    """Receives dense floating point data from PyInPort and stores it in a Var."""
    var: np.ndarray = LavaPyType(np.ndarray, float)
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        self.var = self.in_port.recv()


@implements(proc=Recv, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLoihiFixedPointRecvProcessModel(PyLoihiProcessModel):
    """Receives dense fixed point data from PyInPort and stores it in a Var."""
    var: np.ndarray = LavaPyType(np.ndarray, int)
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    def run_spk(self) -> None:
        self.var = self.in_port.recv()


class TestSyncInjector(unittest.TestCase):
    def test_init(self):
        pass

    def test_invalid_shape(self):
        pass

    def test_invalid_dtype(self):
        pass

    def test_send_data_throws_error_if_runtime_not_running(self):
        pass


class TestAsyncInjector(unittest.TestCase):
    def test_init(self):
        pass

    def test_invalid_shape(self):
        pass

    def test_invalid_size(self):
        pass

    def test_invalid_dtype(self):
        pass

    def test_send_data_throws_error_if_runtime_not_running(self):
        pass


class TestPySyncInjectorModelFloat(unittest.TestCase):
    def test_init(self):
        pass

    def test_send_data(self):
        pass

    def test_run_steps_blocking_num_send_lt_num_steps(self):
        pass

    def test_run_steps_blocking_num_send_eq_num_steps(self):
        pass

    def test_run_steps_blocking_num_send_gt_num_steps(self):
        pass

    def test_run_steps_nonblocking_num_send_lt_num_steps(self):
        pass

    def test_run_steps_nonblocking_num_send_eq_num_steps(self):
        pass

    def test_run_steps_nonblocking_num_send_gt_num_steps(self):
        pass

    def test_run_coninouus(self):
        pass


class TestPyAsyncInjectorModelFloat(unittest.TestCase):
    def test_init(self):
        pass
