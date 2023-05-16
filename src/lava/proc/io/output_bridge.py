import numpy as np
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel, \
    PyAsyncProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort


# TODO : (GK) We need a better convention for naming ProcessModels

class OutputBridge(AbstractProcess):
    """Process that gets data out of Lava.

    This Process exposes a method, receive_data, which can be used to
    effectively relay data from the OutPort to which the InPort of this
    Process is connected to outside Lava.

    Parameters
    ----------
    shape : tuple
        Shape of the InPort of the Process.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._recv_pipe, send_pipe = mp.Pipe(duplex=False)
        super().__init__(shape=shape, send_pipe=send_pipe)

        self.in_port = InPort(shape=shape)

    def receive_data(self) -> np.ndarray:
        """Receive data from the ProcessModel and return it.

        The data received from the ProcessModel is the data coming from the
        OutPort to which the InPort of this Process is connected.

        Returns
        ----------
        data : np.ndarray
            Data got from Lava.
        """
        return self._recv_pipe.recv()


class AbstractPyLoihiOutputBridgeProcessModel(PyLoihiProcessModel):
    """Abstract PyLoihiProcessModel for the OutputBridge Process."""
    in_port = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._send_pipe: mp.connection.PipeConnection = \
            self.proc_params["send_pipe"]

    def run_spk(self) -> None:
        self._send_pipe.send(self.in_port.recv())


@implements(proc=OutputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLoihiFloatingPointOutputBridgeProcessModel(AbstractPyLoihiOutputBridgeProcessModel):
    """Floating point PyLoihiProcessModel for the OutputBridge Process."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)


@implements(proc=OutputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLoihiFixedPointOutputBridgeProcessModel(AbstractPyLoihiOutputBridgeProcessModel):
    """Fixed point PyLoihiProcessModel for the OutputBridge Process."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)


class AbstractPyAsyncOutputBridgeProcessModel(PyLoihiProcessModel):
    """Abstract PyAsyncProcessModel for the OutputBridge Process."""
    in_port = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._send_pipe: mp.connection.PipeConnection = \
            self.proc_params["send_pipe"]

    def run_spk(self) -> None:
        self._send_pipe.send(self.in_port.recv())


@implements(proc=OutputBridge, protocol=AsyncProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLoihiFloatingPointOutputBridgeProcessModel(AbstractPyAsyncOutputBridgeProcessModel):
    """Floating point PyAsyncProcessModel for the OutputBridge Process."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)


@implements(proc=OutputBridge, protocol=AsyncProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLoihiFixedPointOutputBridgeProcessModel(AbstractPyAsyncOutputBridgeProcessModel):
    """Fixed point PyAsyncProcessModel for the OutputBridge Process."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)


