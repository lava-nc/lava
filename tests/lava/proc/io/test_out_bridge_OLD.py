import numpy as np
import unittest
import multiprocessing as mp
import threading
import time

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous

from lava.proc.io.out_bridge import AsyncOutputBridge, SyncOutputBridge


class Send(AbstractProcess):
    def __init__(self, shape):
        super().__init__(shape=shape)

        self.out_port = OutPort(shape=shape)


@implements(proc=Send, protocol=LoihiProtocol)
@requires(CPU)
class PyLoihiFloatingPointSendProcessModel(PyLoihiProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        print(f"Send run_spk {self.time_step} before send...")
        self.out_port.send(np.random.randint(100, size=self.out_port.shape))
        print(f"Send run_spk {self.time_step} after send!")


class TestAsyncOutputBridge(unittest.TestCase):
    def test_run_steps_blocking(self):
        np.random.seed(0)

        num_steps = 300
        num_receive = num_steps
        data_shape = (1,)

        send = Send(shape=data_shape)
        output_bridge = AsyncOutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=True)
        run_cfg = Loihi2SimCfg()

        def thread_2_fn():
            for i in range(num_receive):
                received_data = output_bridge.receive_data()
                print(f"{i + 1} after receive_data!")

        thread = threading.Thread(target=thread_2_fn, daemon=True)
        thread.start()

        output_bridge.run(condition=run_condition, run_cfg=run_cfg)

        print("Joining...")

        thread.join()

        print("Stopping...")

        output_bridge.stop()

    def test_run_steps_blocking_2(self):
        np.random.seed(0)

        num_steps = 300
        num_receive = num_steps - 1
        data_shape = (1,)

        send = Send(shape=data_shape)
        output_bridge = AsyncOutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=True)
        run_cfg = Loihi2SimCfg()

        def thread_2_fn():
            for i in range(num_receive):
                received_data = output_bridge.receive_data()
                print(f"{i + 1} after receive_data!")

        thread = threading.Thread(target=thread_2_fn, daemon=True)
        thread.start()

        output_bridge.run(condition=run_condition, run_cfg=run_cfg)

        print("Joining...")

        thread.join()

        print("Stopping...")

        output_bridge.stop()

    def test_run_steps_blocking_3(self):
        np.random.seed(0)

        num_steps = 300
        num_receive = num_steps + 100
        data_shape = (1,)

        send = Send(shape=data_shape)
        output_bridge = AsyncOutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=True)
        run_cfg = Loihi2SimCfg()

        def thread_2_fn():
            for i in range(num_receive):
                received_data = output_bridge.receive_data()
                print(f"{i + 1} after receive_data!")

        thread = threading.Thread(target=thread_2_fn, daemon=True)
        thread.start()

        output_bridge.run(condition=run_condition, run_cfg=run_cfg)

        print("Joining...")

        thread.join()

        print("Stopping...")

        output_bridge.stop()

    def test_run_steps_non_blocking(self):
        np.random.seed(0)

        num_steps = 300
        num_receive = num_steps
        data_shape = (1,)

        send = Send(shape=data_shape)
        output_bridge = AsyncOutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg()

        output_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_receive):
            received_data = output_bridge.receive_data()
            print(f"{i + 1} after receive_data!")

        print("Stopping...")

        output_bridge.stop()

    def test_run_steps_non_blocking_1(self):
        np.random.seed(0)

        num_steps = 300
        num_receive = num_steps - 1
        data_shape = (1,)

        send = Send(shape=data_shape)
        output_bridge = AsyncOutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg()

        output_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_receive):
            received_data = output_bridge.receive_data()
            print(f"{i + 1} after receive_data!")

        print("Stopping...")

        output_bridge.stop()

    def test_run_steps_non_blocking_2(self):
        np.random.seed(0)

        num_steps = 300
        num_receive = num_steps + 100
        data_shape = (1,)

        send = Send(shape=data_shape)
        output_bridge = AsyncOutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg()

        output_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_receive):
            received_data = output_bridge.receive_data()
            print(f"{i + 1} after receive_data!")

        print("Stopping...")

        output_bridge.stop()

    def test_run_continuous(self):
        np.random.seed(0)

        num_receive = 300
        data_shape = (1,)

        send = Send(shape=data_shape)
        output_bridge = AsyncOutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunContinuous()
        run_cfg = Loihi2SimCfg()

        output_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_receive):
            received_data = output_bridge.receive_data()

        output_bridge.stop()

    def test_run_continuous(self):
        np.random.seed(0)

        num_receive = 300
        data_shape = (1,)

        send = Send(shape=data_shape)
        output_bridge = AsyncOutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunContinuous()
        run_cfg = Loihi2SimCfg()

        output_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_receive):
            received_data = output_bridge.receive_data()
            print(f"{i + 1} after receive_data!")

        print("Stopping...")

        output_bridge.stop()

    def test_run_continuous_1(self):
        np.random.seed(0)

        num_receive = 300
        data_shape = (1,)

        send = Send(shape=data_shape)
        output_bridge = AsyncOutputBridge(shape=data_shape)

        send.out_port.connect(output_bridge.in_port)

        run_condition = RunContinuous()
        run_cfg = Loihi2SimCfg()

        output_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_receive):
            received_data = output_bridge.receive_data()

        time.sleep(1)

        output_bridge.stop()


# class TestSyncOutputBridge(unittest.TestCase):
#     def test_run_steps_blocking(self):
#         num_steps = 10
#         data_shape = (1,)
#         send_data = np.ones(data_shape)
#
#         input_bridge = SyncInputBridge(shape=data_shape)
#         recv = Recv(shape=data_shape)
#
#         input_bridge.out_port.connect(recv.in_port)
#
#         run_condition = RunSteps(num_steps=num_steps, blocking=True)
#         run_cfg = Loihi2SimCfg()
#
#         input_bridge.run(condition=run_condition, run_cfg=run_cfg)
#
#         input_bridge.send_data(send_data)
#
#         input_bridge.stop()
#
#     def test_run_steps_non_blocking(self):
#         num_steps = 10
#         data_shape = (1,)
#         send_data = np.ones(data_shape)
#
#         input_bridge = SyncInputBridge(shape=data_shape)
#         recv = Recv(shape=data_shape)
#
#         input_bridge.out_port.connect(recv.in_port)
#
#         run_condition = RunSteps(num_steps=num_steps, blocking=False)
#         run_cfg = Loihi2SimCfg()
#
#         input_bridge.run(condition=run_condition, run_cfg=run_cfg)
#
#         input_bridge.send_data(send_data)
#
#         input_bridge.stop()
#
#     def test_run_continuous(self):
#         data_shape = (1,)
#         send_data = np.ones(data_shape)
#
#         input_bridge = SyncInputBridge(shape=data_shape)
#         recv = Recv(shape=data_shape)
#
#         input_bridge.out_port.connect(recv.in_port)
#
#         run_condition = RunContinuous()
#         run_cfg = Loihi2SimCfg()
#
#         input_bridge.run(condition=run_condition, run_cfg=run_cfg)
#
#         input_bridge.send_data(send_data)
#
#         input_bridge.stop()


if __name__ == "__main__":
    unittest.main()
