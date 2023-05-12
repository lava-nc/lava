import numpy as np
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort


# TODO : (GK) Is this really a good name ?
class InputBridge(AbstractProcess):
    """Process that gets data into Lava.

    Parameters
    ----------
    out_shape : tuple
        Shape of the OutPort of the Process.
    """

    def __init__(self, out_shape: tuple[int, ...]) -> None:
        # TODO : (GK) I don't like having code before the call to super()... Is there a better way to do it ?
        recv_pipe, self._send_pipe = mp.Pipe(duplex=False)
        super().__init__(out_shape=out_shape, recv_pipe=recv_pipe)

        self.out_port = OutPort(shape=out_shape)

    def send_data(self, data: np.ndarray) -> None:
        """Send data to the ProcessModel.

        The data sent to the ProcessModel will be received by it, and then sent by the ProcessModel through the OutPort.

        Parameters
        ----------
        data : np.ndarray
            Data to get into Lava.
        """
        if data.shape != self.out_port.shape:
            raise ValueError(f"Shape of data to send must be the same as the shape of the OutPort. "
                             f"{data.shape=} != out_port.shape={self.out_port.shape}.")

        self._send_pipe.send(data)


# TODO : (GK) What's the best convention for naming ProcessModels ?
@implements(proc=InputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLoihiFloatingPointInputBridgeProcessModel(PyLoihiProcessModel):
    """PyLoihiFloatingPointProcessModel for the InputBridge Process."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._recv_pipe: mp.connection.PipeConnection = self.proc_params["recv_pipe"]

    def run_spk(self) -> None:
        self.out_port.send(self._recv_pipe.recv())


# TODO : (GK) What's the best convention for naming ProcessModels ?
@implements(proc=InputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLoihiFixedPointInputBridgeProcessModel(PyLoihiProcessModel):
    """PyLoihiFixedPointProcessModel for the InputBridge Process."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._recv_pipe: mp.connection.PipeConnection = self.proc_params["recv_pipe"]

    def run_spk(self) -> None:
        self.out_port.send(self._recv_pipe.recv())
