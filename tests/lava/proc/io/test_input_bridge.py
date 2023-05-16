import numpy as np
import unittest
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

from lava.proc.io.input_bridge import InputBridge, \
    PyLoihiFloatingPointInputBridgeProcessModel, \
    PyLoihiFixedPointInputBridgeProcessModel


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


class TestInputBridge(unittest.TestCase):
    def test_init(self):
        out_shape = (1, )

        input_bridge = InputBridge(shape=out_shape)

        self.assertIsInstance(input_bridge, InputBridge)
        self.assertIsInstance(input_bridge._send_pipe, mp.connection.PipeConnection)
        self.assertIsInstance(input_bridge.proc_params["recv_pipe"], mp.connection.PipeConnection)
        self.assertEqual(input_bridge.out_port.shape, out_shape)

    # TODO : Is this test really necessary ?
    # def test_invalid_shape(self):
    #     out_shape = (1.5, )
    #
    #     with self.assertRaises(ValueError):
    #         InputBridge(out_shape=out_shape)

    def test_send_data(self):
        out_shape = (1,)

        input_bridge = InputBridge(shape=out_shape)

        send_data = np.ones(out_shape)
        input_bridge.send_data(send_data)

        recv_data = input_bridge.proc_params["recv_pipe"].recv()

        np.testing.assert_equal(recv_data, send_data)

    def test_send_data_shape_different_from_out_shape(self):
        out_shape = (1,)

        input_bridge = InputBridge(shape=out_shape)

        data_shape = (1, 2)
        send_data = np.ones(data_shape)

        with self.assertRaises(ValueError):
            input_bridge.send_data(send_data)

    def test_send_data_multiple_times_before_receiving(self):
        # limit for (1, ) = 54
        # limit for (2, ) = 52
        # limit for (3, ) = 51

        # limit for (4, ) = 50
        # limit for (4,1) = 49
        # limit for (2,2) = 49

        # limit for (50, ) = 23
        # limit for (50, 1) = 23
        # limit for (5, 10) = 23

        # limit for (100, ) = 14
        # limit for (500, ) = 3
        # limit for (986, ) = 2
        # limit for (2010, ) = 1
        # limit for (2011, ) = 0

        num_consecutive_send_before_receive = 50
        out_shape = (4, )

        input_bridge = InputBridge(shape=out_shape)

        complete_send_data = [np.full(out_shape, i) for i in range(num_consecutive_send_before_receive)]

        for send_data in complete_send_data:
            input_bridge.send_data(send_data)

        complete_recv_data = []

        for _ in range(num_consecutive_send_before_receive):
            complete_recv_data.append(input_bridge.proc_params["recv_pipe"].recv())

        np.testing.assert_equal(complete_recv_data, complete_send_data)


class TestInputBridgeFloatingPointProcessModel(unittest.TestCase):
    def test_init(self):
        out_shape = (1, )
        recv_pipe, _ = mp.Pipe(duplex=False)
        proc_params = {
            "out_shape": out_shape,
            "recv_pipe": recv_pipe
        }

        process_model = PyLoihiFloatingPointInputBridgeProcessModel(proc_params=proc_params)

        self.assertIsInstance(process_model, PyLoihiFloatingPointInputBridgeProcessModel)
        self.assertIsInstance(process_model._recv_pipe, mp.connection.PipeConnection)

    def test_run(self):
        num_steps = 1
        data_shape = (1,)
        send_data = np.ones(data_shape)

        input_bridge = InputBridge(shape=data_shape)
        recv = Recv(shape=data_shape)

        input_bridge.out_port.connect(recv.in_port)

        # run_condition has to be non-blocking in order to get past the call to run
        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg(select_tag="floating_pt")

        input_bridge.run(condition=run_condition, run_cfg=run_cfg)

        input_bridge.send_data(send_data)

        input_bridge.wait()

        recv_data = recv.var.get()

        input_bridge.stop()

        np.testing.assert_equal(recv_data, send_data)


class TestInputBridgeFixedPointProcessModel(unittest.TestCase):
    def test_init(self):
        out_shape = (1, )
        recv_pipe, _ = mp.Pipe(duplex=False)
        proc_params = {
            "out_shape": out_shape,
            "recv_pipe": recv_pipe
        }

        process_model = PyLoihiFixedPointInputBridgeProcessModel(proc_params=proc_params)

        self.assertIsInstance(process_model, PyLoihiFixedPointInputBridgeProcessModel)
        self.assertIsInstance(process_model._recv_pipe, mp.connection.PipeConnection)

    def test_run(self):
        num_steps = 1
        data_shape = (1,)
        send_data = np.ones(data_shape, dtype=int)

        input_bridge = InputBridge(shape=data_shape)
        recv = Recv(shape=data_shape)

        input_bridge.out_port.connect(recv.in_port)

        # run_condition has to be non-blocking in order to get past the call to run
        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        input_bridge.run(condition=run_condition, run_cfg=run_cfg)

        input_bridge.send_data(send_data)

        input_bridge.wait()

        recv_data = recv.var.get()

        input_bridge.stop()

        np.testing.assert_equal(recv_data, send_data)

if __name__ == "__main__":
    unittest.main()
