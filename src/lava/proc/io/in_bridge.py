import numpy as np
import time
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel, PyAsyncProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort


class AsyncInputBridge(AbstractProcess):
    def __init__(self, shape):
        pm_pipe, self._p_pipe = mp.Pipe(duplex=False)
        super().__init__(shape=shape, pm_pipe=pm_pipe)

        self.out_port = OutPort(shape=shape)

    def send_data(self, data):
        self._p_pipe.send(data)


@implements(proc=AsyncInputBridge, protocol=AsyncProtocol)
@requires(CPU)
class AsyncProcessModel(PyAsyncProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

        self._pm_pipe = self.proc_params["pm_pipe"]

    def run_async(self) -> None:
        counter_large = 0
        counter_small = 0
        while True:
            counter_large += 1

            if counter_large <= 1000:
                print(f"AsyncProcessModel run_async before poll "
                      f"{counter_large}")

            # print(f"AsyncProcessModel run_async before poll "
            #       f"{counter_large}")

            if self._pm_pipe.poll():
                counter_small += 1
                recv_data = self._pm_pipe.recv()
                self.out_port.send(recv_data)
                print(f"AsyncProcessModel run_async sent! {counter_small}")

            if self.check_for_stop_cmd():
                print("AsyncProcessModel run_async found stop cmd!")
                return


class SyncInputBridge(AbstractProcess):
    def __init__(self, shape):
        pm_pipe, self._p_pipe = mp.Pipe(duplex=False)
        super().__init__(shape=shape, pm_pipe=pm_pipe)

        self.out_port = OutPort(shape=shape)

    def send_data(self, data):
        self._p_pipe.send(data)


@implements(proc=SyncInputBridge, protocol=LoihiProtocol)
@requires(CPU)
class SyncProcessModel(PyLoihiProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

        self._pm_pipe = self.proc_params["pm_pipe"]

    def run_spk(self) -> None:
        self.out_port.send(self._pm_pipe.recv())
