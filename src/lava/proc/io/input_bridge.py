import numpy as np
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel, \
    PyAsyncProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort


# TODO : (GK) We need a better convention for naming ProcessModels

class InputBridge(AbstractProcess):
    """Process that gets data into Lava.

    This Process exposes a method, send_data, which can be used to effectively
    relay data from outside Lava to the InPort connected to the OutPort of this
    Process.

    Parameters
    ----------
    shape : tuple
        Shape of the OutPort of the Process.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        recv_pipe, self._send_pipe = mp.Pipe(duplex=False)
        super().__init__(shape=shape, recv_pipe=recv_pipe)

        self.out_port = OutPort(shape=shape)

    def send_data(self, data: np.ndarray) -> None:
        """Send data to the ProcessModel.

        The data received by the ProcessModel will then be sent through the
        OutPort and end up at the InPort connected to it.

        Parameters
        ----------
        data : np.ndarray
            Data to get into Lava.
        """
        if data.shape != self.out_port.shape:
            raise ValueError(f"Shape of data to send must be the same as the "
                             f"shape of the OutPort. "
                             f"{data.shape=}. "
                             f"out_port.shape={self.out_port.shape}.")

        self._send_pipe.send(data)


class AbstractPyLoihiInputBridgeProcessModel(PyLoihiProcessModel):
    """Abstract PyLoihiProcessModel for the InputBridge Process."""
    out_port = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._recv_pipe: mp.connection.PipeConnection = \
            self.proc_params["recv_pipe"]

    def run_spk(self) -> None:
        self.out_port.send(self._recv_pipe.recv())


@implements(proc=InputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLoihiFloatingPointInputBridgeProcessModel(AbstractPyLoihiInputBridgeProcessModel):
    """Floating point PyLoihiProcessModel for the InputBridge Process."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)


@implements(proc=InputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLoihiFixedPointInputBridgeProcessModel(AbstractPyLoihiInputBridgeProcessModel):
    """Fixed point PyLoihiProcessModel for the InputBridge Process."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)


class AbstractPyAsyncInputBridgeProcessModel(PyAsyncProcessModel):
    """Abstract PyAsyncProcessModel for the InputBridge Process."""
    out_port = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._recv_pipe: mp.connection.PipeConnection = \
            self.proc_params["recv_pipe"]

    def run_async(self):
        while True:
            self.out_port.send(self._recv_pipe.recv())

            if self.check_for_stop_cmd():
                return


@implements(proc=InputBridge, protocol=AsyncProtocol)
@requires(CPU)
@tag("floating_pt")
class PyAsyncFloatingPointInputBridgeProcessModel(AbstractPyAsyncInputBridgeProcessModel):
    """Floating point PyAsyncProcessModel for the InputBridge Process."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)


@implements(proc=InputBridge, protocol=AsyncProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyAsyncFixedPointInputBridgeProcessModel(AbstractPyAsyncInputBridgeProcessModel):
    """Fixed point PyAsyncProcessModel for the InputBridge Process."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
