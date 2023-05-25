import numpy as np
from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort

# AsyncInput assumes that the sensor is not synced
# it still runs in Loihiprotocol (synced with other processes)
from lava.magma.runtime.message_infrastructure.multiprocessing import MultiProcessing

# Inheritance Structure
# Processes
#               Injector
# AsyncInjector            SyncInjector

# Process Models
#               PyInjectorModel
# PyAsyncInjectorModel       PySyncInjectorModel
# fixed floating             fixed floating

class Injector(AbstractProcess):
    def __init__(self, shape, dtype, size):
        super().__init__(shape=shape)
        mp = MultiProcessing()
        mp.start()
        self.channel = PyPyChannel(message_infrastructure=mp,
                                   src_name="source",
                                   dst_name="destination",
                                   shape=shape,
                                   dtype=dtype,
                                   size=size)
        self.proc_params["dst_port"] = self.channel.dst_port
        self.src_port = self.channel.src_port
        self.src_port.start()
        self.out_port = OutPort(shape=shape)

    def send_data(self, data):
        self.src_port.send(data)


class AsyncInjector(Injector):
    pass


class SyncInjector(Injector):
    def __init__(self, shape, dtype):
        # Ensure there is always just one object in the PyPyChannel
        super().__init__(shape=shape, dtype=dtype, size=1)

class PyInjectorModel(PyLoihiProcessModel):
    out_port = None

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.dst_port = self.proc_params["dst_port"]
        self.dst_port.start()
        self.shape = self.proc_params["shape"]

class PyAsyncInjectorModel(PyInjectorModel):
    # Implementing Default FiFO behavior
    def run_spk(self):
        data = np.zeros(self.shape)
        if self.dst_port.probe():
            data = self.dst_port.recv()
        self.out_port.send(data)

    # ADD different modes of synchronizing later
    # def run_spk(self) -> None:
    #     data = np.zeros(self.shape)
    #     # Get number of elements in queue right now
    #     # Changes as sensor sends more data
    #     elements_in_q = self.dst_port._queue._qsize()
    #     for _ in range(elements_in_q):
    #         data += self.dst_port.recv()
    #     self.out_port.send(data)


class PySyncInjectorModel(PyInjectorModel):
    def run_spk(self):
        # Block unless data arrives
        data = self.dst_port.recv()
        self.out_port.send(data)

@implements(proc=SyncInjector, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PySyncInjectorModelFloat(PySyncInjectorModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)


@implements(proc=AsyncInjector, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyAsyncInjectorModelFloat(PyAsyncInjectorModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
