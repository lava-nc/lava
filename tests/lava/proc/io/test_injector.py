import numpy as np
import unittest
import threading
import time
from time import sleep

from lava.magma.compiler.channels.pypychannel import PyPyChannel, CspRecvPort, CspSendPort
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

from lava.proc.io.in_bridge import AsyncInjector


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


class TestAsyncInjector(unittest.TestCase):
    def test_init(self):
        out_shape = (1,)
        size = 10
        dtype = float
        injector = AsyncInjector(shape=out_shape, dtype=dtype, size=size)
        self.assertIsInstance(injector, AsyncInjector)
        self.assertIsInstance(injector._channel, PyPyChannel)
        self.assertIsInstance(injector.proc_params["dst_port"], CspRecvPort)
        self.assertIsInstance(injector._src_port, CspSendPort)
        self.assertEqual(injector.out_port.shape, out_shape)

    def test_invalid_shape(self):
        out_shape = (1.5,)
        size = 10
        with self.assertRaises(TypeError):
            AsyncInjector(shape=out_shape, dtype=float, size=size)

        out_shape = (-1,)
        with self.assertRaises(ValueError):
            AsyncInjector(shape=out_shape, dtype=float, size=size)

        out_shape = 4
        with self.assertRaises(TypeError):
            AsyncInjector(shape=out_shape, dtype=float, size=size)

    def test_invalid_size(self):
        out_shape = (1,)
        dtype = float
        size = 0.5
        with self.assertRaises(TypeError):
            AsyncInjector(shape=out_shape, dtype=float, size=size)

        size = -5
        with self.assertRaises(ValueError):
            AsyncInjector(shape=out_shape, dtype=float, size=size)

    # Add Input Handling in Process
    #def test_invalid_dtype(self):
    #    dtype = "floati"
    #    AsyncInjector(shape=(1,), dtype=dtype)

    def test_send_data_throws_error_if_runtime_not_running(self):
        shape = (1,)
        size = 10
        data = np.ones(shape)
        injector = AsyncInjector(shape=shape, dtype=float, size=size)
        with self.assertRaises(Exception):
            injector.send_data(data)


class TestPyAsyncInjectorModelFloat(unittest.TestCase):
    def test_init(self):
        pass
