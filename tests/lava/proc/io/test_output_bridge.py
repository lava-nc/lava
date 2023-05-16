import numpy as np
import unittest
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

from lava.proc.io.output_bridge import OutputBridge, \
    PyLoihiFloatingPointOutputBridgeProcessModel, \
    PyLoihiFixedPointOutputBridgeProcessModel

class Send(AbstractProcess):
    """Process that sends arbitrary dense data.

    Parameters
    ----------
    shape: tuple
        Shape of the OutPort.
    data: np.ndarray
        Data to send through the OutPort.
    """
    def __init__(self, shape: tuple[...], data: np.ndarray) -> None:
        super().__init__(shape=shape, data=data)

        self.out_port = OutPort(shape=shape)


@implements(proc=Send, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLoihiFloatingPointSendProcessModel(PyLoihiProcessModel):
    """Sends dense floating point data to PyOutPort."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict):
        super().__init__(proc_params)
        self._data = proc_params["data"]

    def run_spk(self) -> None:
        self.out_port.send(data=self._data)


@implements(proc=Send, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLoihiFixedPointSendProcessModel(PyLoihiProcessModel):
    """Sends dense fixed point data to PyOutPort."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params: dict):
        super().__init__(proc_params)
        self._data = proc_params["data"]

    def run_spk(self) -> None:
        self.out_port.send(data=self._data)



class TestOutputBridge(unittest.TestCase):
    def test_init(self):
        in_shape = (1, )

        output_bridge = OutputBridge(shape=in_shape)

        self.assertIsInstance(output_bridge, OutputBridge)
        self.assertIsInstance(output_bridge._recv_pipe, mp.connection.PipeConnection)
        self.assertIsInstance(output_bridge.proc_params["send_pipe"], mp.connection.PipeConnection)
        self.assertEqual(output_bridge.in_port.shape, in_shape)

    # TODO : Is this test really necessary ?
    # def test_invalid_shape(self):
    #     in_shape = (1.5, )
    #
    #     with self.assertRaises(ValueError):
    #         OutputBridge(shape=in_shape)

    def test_send_data(self):
        in_shape = (1,)

        output_bridge = OutputBridge(shape=in_shape)

        send_data = np.ones(in_shape)
        output_bridge.proc_params["send_pipe"].send(send_data)

        recv_data = output_bridge.receive_data()

        np.testing.assert_equal(recv_data, send_data)

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
        in_shape = (4, )

        output_bridge = OutputBridge(shape=in_shape)

        complete_send_data = [np.full(in_shape, i) for i in range(num_consecutive_send_before_receive)]

        for send_data in complete_send_data:
            output_bridge.proc_params["send_pipe"].send(send_data)

        complete_recv_data = []

        for _ in range(num_consecutive_send_before_receive):
            complete_recv_data.append(output_bridge.receive_data())

        np.testing.assert_equal(complete_recv_data, complete_send_data)


class TestInputBridgePyLoihiFloatingPointProcessModel(unittest.TestCase):
    def test_init(self):
        in_shape = (1, )
        _, send_pipe = mp.Pipe(duplex=False)
        proc_params = {
            "out_shape": in_shape,
            "send_pipe": send_pipe
        }

        process_model = PyLoihiFloatingPointOutputBridgeProcessModel(proc_params=proc_params)

        self.assertIsInstance(process_model, PyLoihiFloatingPointOutputBridgeProcessModel)
        self.assertIsInstance(process_model._send_pipe, mp.connection.PipeConnection)

    def test_run(self):
        num_steps = 1
        data_shape = (1,)
        send_data = np.ones(data_shape)

        send = Send(shape=data_shape, data=send_data)
        output_bridge = OutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg(select_tag="floating_pt")

        send.run(condition=run_condition, run_cfg=run_cfg)

        recv_data = output_bridge.receive_data()

        send.stop()

        np.testing.assert_equal(recv_data, send_data)


class TestInputBridgePyLoihiFixedPointProcessModel(unittest.TestCase):
    def test_init(self):
        in_shape = (1, )
        _, send_pipe = mp.Pipe(duplex=False)
        proc_params = {
            "out_shape": in_shape,
            "send_pipe": send_pipe
        }

        process_model = PyLoihiFixedPointOutputBridgeProcessModel(proc_params=proc_params)

        self.assertIsInstance(process_model, PyLoihiFixedPointOutputBridgeProcessModel)
        self.assertIsInstance(process_model._send_pipe, mp.connection.PipeConnection)

    def test_run(self):
        num_steps = 1
        data_shape = (1,)
        send_data = np.ones(data_shape, dtype=int)

        send = Send(shape=data_shape, data=send_data)
        output_bridge = OutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunSteps(num_steps=num_steps)
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

        send.run(condition=run_condition, run_cfg=run_cfg)

        recv_data = output_bridge.receive_data()

        send.stop()

        np.testing.assert_equal(recv_data, send_data)

if __name__ == "__main__":
    unittest.main()
