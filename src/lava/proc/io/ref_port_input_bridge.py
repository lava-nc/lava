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

class RefPortInputBridge(AbstractProcess):
    """Process that gets data into Lava.

    This Process exposes a method, send_data, which can be used to effectively
    relay data from outside Lava to the Var connected to the RefPort of this
    Process.

    Parameters
    ----------
    shape : tuple
        Shape of the RefPort of the Process.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        recv_pipe, self._send_pipe = mp.Pipe(duplex=False)
        super().__init__(shape=shape, recv_pipe=recv_pipe)

        self.ref_port = RefPort(shape=shape)

    def send_data(self, data: np.ndarray) -> None:
        """Send data to the ProcessModel.

        The data received by the ProcessModel will then be sent through the
        RefPort and end up at the Var connected to it.

        Parameters
        ----------
        data : np.ndarray
            Data to get into Lava.
        """
        if data.shape != self.ref_port.shape:
            raise ValueError(f"Shape of data to send must be the same as the "
                             f"shape of the RefPort. "
                             f"{data.shape=}. "
                             f"ref_port.shape={self.ref_port.shape}.")

        self._send_pipe.send(data)


class AbstractPyLoihiRefPortInputBridgeProcessModel(PyLoihiProcessModel):
    """Abstract PyLoihiProcessModel for the RefPortInputBridge Process."""
    ref_port = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._recv_pipe: mp.connection.PipeConnection = \
            self.proc_params["recv_pipe"]

    # TODO : Should this really happen only in post_mgmt?

    def post_guard(self) -> bool:
        return True

    def run_post_mgmt(self) -> None:
        self.ref_port.write(self._recv_pipe.recv())


@implements(proc=RefPortInputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLoihiFloatingPointRefPortInputBridgeProcessModel(AbstractPyLoihiRefPortInputBridgeProcessModel):
    """Floating point PyLoihiProcessModel for the RefPortInputBridge Process."""
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)


@implements(proc=RefPortInputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLoihiFixedPointRefPortInputBridgeProcessModel(AbstractPyLoihiRefPortInputBridgeProcessModel):
    """Fixed point PyLoihiProcessModel for the RefPortInputBridge Process."""
    ref_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)


class AbstractPyAsyncRefPortInputBridgeProcessModel(PyAsyncProcessModel):
    """Abstract PyAsyncProcessModel for the RefPortInputBridge Process."""
    ref_port = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._recv_pipe: mp.connection.PipeConnection = \
            self.proc_params["recv_pipe"]

    def run_async(self):
        while True:
            self.ref_port.write(self._recv_pipe.recv())

            if self.check_for_stop_cmd():
                return


@implements(proc=RefPortInputBridge, protocol=AsyncProtocol)
@requires(CPU)
@tag("floating_pt")
class PyAsyncFloatingPointRefPortInputBridgeProcessModel(AbstractPyAsyncRefPortInputBridgeProcessModel):
    """Floating point PyAsyncProcessModel for the RefPortInputBridge Process."""
    out_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)


@implements(proc=RefPortInputBridge, protocol=AsyncProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyAsyncFixedPointRefPortInputBridgeProcessModel(AbstractPyAsyncRefPortInputBridgeProcessModel):
    """Fixed point PyAsyncProcessModel for the RefPortInputBridge Process."""
    out_port: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)
