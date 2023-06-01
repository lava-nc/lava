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


# Inheritance Structure -> removed sync for now
# Processes
#               (Injector)
# AsyncInjector            (SyncInjector)

# Process Models
#               (PyInjectorModel)
# PyAsyncInjectorModel       (PySyncInjectorModel)
# fixed floating             (fixed floating)

class AsyncInjector(AbstractProcess):
    def __init__(self, shape, dtype, size):
        super().__init__(shape=shape)
        self._validate_shape(shape)
        self._validate_size(size)
        self.size = size
        mp = MultiProcessing()
        mp.start()
        self._channel = PyPyChannel(message_infrastructure=mp,
                                    src_name="source",
                                    dst_name="destination",
                                    shape=shape,
                                    dtype=dtype,
                                    size=self.size)
        self.proc_params["dst_port"] = self._channel.dst_port
        self._src_port = self._channel.src_port
        self._dst_port = self._channel.dst_port
        self._src_port.start()
        self.out_port = OutPort(shape=shape)

    def send_data(self, data):
        # First ensure runtime is running
        if not self.runtime._is_running:
            raise Exception("Data can only be sent once the runtime has started.")
        # Check if queue is full
        elements_in_q = self._dst_port._queue.qsize()
        if elements_in_q == self.size:
            # Queue is full, we need to discard one element in the queue
            self._dst_port.recv()
        self._src_port.send(data)

    def _validate_shape(self, shape):
        # If <shape> is not yet a tuple, raise a TypeError
        if not isinstance(shape, tuple):
            raise TypeError("<shape> must be of type int or tuple(int)")

        # Check whether all elements in the tuple are of type int
        # and positive
        for s in shape:
            if not isinstance(s, (int, np.integer)):
                raise TypeError("all elements of <shape> must be of type int")
            if s < 0:
                raise ValueError("all elements of <shape> must be greater "
                                 "than zero")

    def _validate_size(self, size):
        if not isinstance(size, int):
            raise TypeError("size must be positive and of type int")
        if size <= 0:
            raise ValueError("size must be positive and of type int")


class PyAsyncInjectorModel(PyLoihiProcessModel):
    out_port = None

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self._dst_port = self.proc_params["dst_port"]
        self._dst_port.start()
        self.shape = self.proc_params["shape"]


@implements(proc=AsyncInjector, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyAsyncInjectorModelFloat(PyAsyncInjectorModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self) -> None:
        data = np.zeros(self.shape)
        # Get number of elements in queue right now
        # Changes as sensor sends more data
        elements_in_q = self._dst_port._queue._qsize()
        for _ in range(elements_in_q):
            data += self._dst_port.recv()
        self.out_port.send(data)


@implements(proc=AsyncInjector, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyAsyncInjectorModelFixed(PyAsyncInjectorModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.data_bitwidth = 32
        self.max_data_val = 2 ** (self.data_bitwidth-1) - 1
        self.min_data_val = - 2 ** (self.data_bitwidth-1)

    def run_spk(self) -> None:
        data = np.zeros(self.shape, np.int32)
        # Get number of elements in queue right now
        # Changes as sensor sends more data
        elements_in_q = self._dst_port._queue._qsize()
        for _ in range(elements_in_q):
            data += self._dst_port.recv()
        data = np.clip(data, self.min_data_val, self.max_data_val)
        self.out_port.send(np.int32(data))
