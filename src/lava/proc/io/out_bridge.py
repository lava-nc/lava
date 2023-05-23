import numpy as np
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel, PyAsyncProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort


class AsyncOutputBridge(AbstractProcess):
    def __init__(self, shape):
        self._p_pipe, pm_pipe = mp.Pipe(duplex=False)
        super().__init__(shape=shape, pm_pipe=pm_pipe)

        self.in_port = InPort(shape=shape)

    def receive_data(self):
        return self._p_pipe.recv()


@implements(proc=AsyncOutputBridge, protocol=AsyncProtocol)
@requires(CPU)
class AsyncProcessModel(PyAsyncProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._pm_pipe = self.proc_params["pm_pipe"]

    def run_async(self) -> None:
        while True:
            if self.in_port.probe():
                self._pm_pipe.send(self.in_port.recv())

            if self.check_for_stop_cmd():
                return


@implements(proc=AsyncOutputBridge, protocol=AsyncProtocol)
@requires(CPU)
class AsyncProcessModel(PyAsyncProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._pm_pipe = self.proc_params["pm_pipe"]

    def run_async(self) -> None:
        counter_large = 0
        counter_small = 0
        while True:
            counter_large += 1

            if counter_large <= 1000:
                print(f"AsyncProcessModel run_async before probe"
                      f"{counter_large}")

            # print(f"AsyncProcessModel run_async before probe"
            #       f"{counter_large}")

            # self._pm_pipe.send(self.in_port.recv())
            # print(f"AsyncProcessModel run_async received! {counter_large}")

            if self.in_port.probe():
                counter_small += 1
                self._pm_pipe.send(self.in_port.recv())
                print(f"AsyncProcessModel run_async received! {counter_small}")

            if self.check_for_stop_cmd():
                print("AsyncProcessModel run_async found stop cmd!")
                return


class SyncOutputBridge(AbstractProcess):
    def __init__(self, shape):
        self._p_pipe, pm_pipe = mp.Pipe(duplex=False)
        super().__init__(shape=shape, pm_pipe=pm_pipe)

        self.in_port = InPort(shape=shape)

    def receive_data(self) -> np.ndarray:
        return self._p_pipe.recv()


@implements(proc=SyncOutputBridge, protocol=LoihiProtocol)
@requires(CPU)
class SyncProcessModel(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._pm_pipe = self.proc_params["pm_pipe"]

    def run_spk(self) -> None:
        self._pm_pipe.send(self.in_port.recv())


