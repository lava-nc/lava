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

# AsyncInput assumes that the sensor is not synced
# it still runs in Loihiprotocol (synced with other processes)
class AsyncInputBridge(AbstractProcess):
    def __init__(self, shape):
        self.q = mp.Queue()
        super().__init__(shape=shape, q=self.q)

        self.out_port = OutPort(shape=shape)

    def send_data(self, data):
        self.q.put(data)


@implements(proc=AsyncInputBridge, protocol=LoihiProtocol)
@requires(CPU)
class AsyncProcessDenseModel(PyLoihiProcessModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.q = self.proc_params["q"]
        self.shape = self.proc_params["shape"]

    def run_spk(self) -> None:
        data = np.zeros(self.shape)
        # Get number of elements in queue right now
        # Changes as sensor sends more data
        elements_in_q = self.q.qsize()
        print(elements_in_q)
        for _ in range(elements_in_q):
            data += self.q.get()
        print("sending: ")
        print(data)
        self.out_port.send(data)


### Sync Input: Assumes the sensor takes care of synchronization
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
