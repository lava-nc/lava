import numpy as np
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import RefPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel, \
    PyAsyncProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyRefPort


# TODO : (GK) We need a better convention for naming ProcessModels

class RefPortOutputBridge(AbstractProcess):
    """Process that gets data out of Lava.

    This Process exposes a method, receive_data, which can be used to
    effectively relay data from the Var to which the RefPort of this
    Process is connected to outside Lava.

    Parameters
    ----------
    shape : tuple
        Shape of the RefPort of the Process.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._recv_pipe, send_pipe = mp.Pipe(duplex=False)
        super().__init__(shape=shape, send_pipe=send_pipe)

        self.ref_port = RefPort(shape=shape)

    def receive_data(self) -> np.ndarray:
        """Receive data from the ProcessModel and return it.

        The data received from the ProcessModel is the data coming from the
        Var to which the RefPort of this Process is connected.

        Returns
        ----------
        data : np.ndarray
            Data got from Lava.
        """
        return self._recv_pipe.recv()


class AbstractPyLoihiRefPortOutputBridgeProcessModel(PyLoihiProcessModel):
    """Abstract PyLoihiProcessModel for the RefPortOutputBridge Process."""
    ref_port = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._send_pipe: mp.connection.PipeConnection = \
            self.proc_params["send_pipe"]

    # TODO : Should this really happen only in post_mgmt?

    def post_guard(self) -> bool:
        return True

    def run_post_mgmt(self) -> None:
        self._send_pipe.send(self.ref_port.read())


@implements(proc=RefPortOutputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLoihiFloatingPointRefPortOutputBridgeProcessModel(AbstractPyLoihiRefPortOutputBridgeProcessModel):
    """Floating point PyLoihiProcessModel for the RefPortOutputBridge Process."""
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)


@implements(proc=RefPortOutputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLoihiFixedPointRefPortOutputBridgeProcessModel(AbstractPyLoihiRefPortOutputBridgeProcessModel):
    """Fixed point PyLoihiProcessModel for the RefPortOutputBridge Process."""
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)


class AbstractPyAsyncRefPortOutputBridgeProcessModel(PyAsyncProcessModel):
    """Abstract PyAsyncProcessModel for the RefPortOutputBridge Process."""
    ref_port = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._send_pipe: mp.connection.PipeConnection = \
            self.proc_params["send_pipe"]

    def run_spk(self) -> None:
        self._send_pipe.send(self.ref_port.read())


@implements(proc=RefPortOutputBridge, protocol=AsyncProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLoihiFloatingPointRefPortOutputBridgeProcessModel(AbstractPyAsyncRefPortOutputBridgeProcessModel):
    """Floating point PyAsyncProcessModel for the RefPortOutputBridge Process."""
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)


@implements(proc=RefPortOutputBridge, protocol=AsyncProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLoihiFixedPointRefPortOutputBridgeProcessModel(AbstractPyAsyncRefPortOutputBridgeProcessModel):
    """Fixed point PyAsyncProcessModel for the RefPortOutputBridge Process."""
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)


