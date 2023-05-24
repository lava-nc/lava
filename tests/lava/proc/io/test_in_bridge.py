import numpy as np
import unittest
import multiprocessing as mp
import threading
import time
from time import sleep
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

from lava.proc.io.in_bridge import AsyncInputBridge, SyncInputBridge


class Recv(AbstractProcess):
    def __init__(self, shape):
        super().__init__(shape=shape)

        self.var = Var(shape=shape, init=0)
        self.in_port = InPort(shape=shape)


@implements(proc=Recv, protocol=LoihiProtocol)
@requires(CPU)
class PyLoihiFloatingPointRecvProcessModel(PyLoihiProcessModel):
    var: np.ndarray = LavaPyType(np.ndarray, float)
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def run_spk(self) -> None:
       # print(f"Recv run_spk {self.time_step} before recv...")
        recv = self.in_port.recv()
      #  print(f"Recv run_spk {self.time_step} after recv!")
        self.var = recv


class TestAsyncInputBridge(unittest.TestCase):
    def test_run_steps_blocking(self):
        num_steps = 300
        data_shape = (1,)

        input_bridge = AsyncInputBridge(shape=data_shape, size=100, dtype=float)
        recv = Recv(shape=data_shape)

        input_bridge.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=True)
        run_cfg = Loihi2SimCfg()

        def thread_2_fn():
            for i in range(100000):
                input_bridge.send_data(np.ones(data_shape))
                if i % 100 == 0:
                    sleep(0.01)

        thread = threading.Thread(target=thread_2_fn, daemon=True)
        thread.start()

        input_bridge.run(condition=run_condition, run_cfg=run_cfg)

        print("Joining...")

        thread.join()

        print("Stopping...")

        input_bridge.stop()

    def test_run_steps_blocking_2(self):
        np.random.seed(0)

        num_steps = 300
        num_send = num_steps - 1
        data_shape = (1,)

        input_bridge = AsyncInputBridge(shape=data_shape)
        recv = Recv(shape=data_shape)

        input_bridge.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=True)
        run_cfg = Loihi2SimCfg()

        def thread_2_fn():
            for i in range(num_send):
                input_bridge.send_data(np.random.randint(100, size=data_shape))

        thread = threading.Thread(target=thread_2_fn, daemon=True)
        thread.start()

        input_bridge.run(condition=run_condition, run_cfg=run_cfg)

        print("Joining...")

        thread.join()

        print("Stopping...")

        input_bridge.stop()

    def test_run_steps_blocking_3(self):
        np.random.seed(0)

        num_steps = 300
        num_send = num_steps + 100
        data_shape = (1,)

        input_bridge = AsyncInputBridge(shape=data_shape)
        recv = Recv(shape=data_shape)

        input_bridge.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=True)
        run_cfg = Loihi2SimCfg()

        def thread_2_fn():
            for i in range(num_send):
                input_bridge.send_data(np.random.randint(100, size=data_shape))

        thread = threading.Thread(target=thread_2_fn, daemon=True)
        thread.start()

        input_bridge.run(condition=run_condition, run_cfg=run_cfg)

        print("Joining...")

        thread.join()

        print("Stopping...")

        input_bridge.stop()

    def test_run_steps_non_blocking(self):
        np.random.seed(0)

        num_steps = 300
        num_send = num_steps
        data_shape = (1,)

        input_bridge = AsyncInputBridge(shape=data_shape)
        recv = Recv(shape=data_shape)

        input_bridge.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg()

        input_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_send):
            input_bridge.send_data(np.random.randint(100, size=data_shape))

        time.sleep(1)

        print("Stopping...")

        input_bridge.stop()

    def test_run_steps_non_blocking_1(self):
        np.random.seed(0)

        num_steps = 300
        num_send = num_steps - 1
        data_shape = (1,)

        input_bridge = AsyncInputBridge(shape=data_shape)
        recv = Recv(shape=data_shape)

        input_bridge.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg()

        input_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_send):
            input_bridge.send_data(np.random.randint(100, size=data_shape))

        time.sleep(1)

        print("Stopping...")

        input_bridge.stop()

    def test_run_steps_non_blocking_2(self):
        np.random.seed(0)

        num_steps = 300
        num_send = num_steps + 100
        data_shape = (1,)

        input_bridge = AsyncInputBridge(shape=data_shape)
        recv = Recv(shape=data_shape)

        input_bridge.out_port.connect(recv.in_port)

        run_condition = RunSteps(num_steps=num_steps, blocking=False)
        run_cfg = Loihi2SimCfg()

        input_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_send):
            input_bridge.send_data(np.random.randint(100, size=data_shape))

        time.sleep(1)

        print("Stopping...")

        input_bridge.stop()

    def test_run_continuous(self):
        np.random.seed(0)

        num_send = 300
        data_shape = (1,)

        input_bridge = AsyncInputBridge(shape=data_shape)
        recv = Recv(shape=data_shape)

        input_bridge.out_port.connect(recv.in_port)

        run_condition = RunContinuous()
        run_cfg = Loihi2SimCfg()

        input_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_send):
            input_bridge.send_data(np.random.randint(100, size=data_shape))

        print("Stopping...")

        input_bridge.stop()

    def test_run_continuous_1(self):
        np.random.seed(0)

        num_send = 300
        data_shape = (1,)

        input_bridge = AsyncInputBridge(shape=data_shape)
        recv = Recv(shape=data_shape)

        input_bridge.out_port.connect(recv.in_port)

        run_condition = RunContinuous()
        run_cfg = Loihi2SimCfg()

        input_bridge.run(condition=run_condition, run_cfg=run_cfg)

        for i in range(num_send):
            input_bridge.send_data(np.random.randint(100, size=data_shape))

        time.sleep(1)

        print("Stopping...")

        input_bridge.stop()


# class TestSyncInputBridge(unittest.TestCase):
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
