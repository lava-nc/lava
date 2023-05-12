import numpy as np
import multiprocessing as mp

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort


# TODO : (GK) Is this really a good name ?
class OutputBridge(AbstractProcess):
    """Process that gets data out of Lava.

    Parameters
    ----------
    shape_in : tuple
        Shape of the InPort of the Process.
    """

    def __init__(self, in_shape: tuple[int, ...]) -> None:
        # TODO : (GK) I don't like having code before the call to super()... Is there a better way to do it ?
        self._recv_pipe, send_pipe = mp.Pipe(duplex=False)
        super().__init__(in_shape=in_shape, send_pipe=send_pipe)

        self.in_port = InPort(shape=in_shape)

    def receive_data(self) -> np.ndarray:
        """Receive data from the ProcessModel.

        The data received from the InPort is sent from the ProcessModel and returned by this method.

        Returns
        ----------
        data : np.ndarray
            Data got from Lava.
        """
        return self._recv_pipe.recv()


# TODO : (GK) What's the best convention for naming ProcessModels ?
@implements(proc=OutputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyLoihiFloatingPointOutputBridgeProcessModel(PyLoihiProcessModel):
    """PyLoihiFloatingPointProcessModel for the OutputBridge Process."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._send_pipe: mp.connection.PipeConnection = self.proc_params["send_pipe"]

    def run_spk(self) -> None:
        self._send_pipe.send(self.in_port.recv())


# TODO : (GK) What's the best convention for naming ProcessModels ?
@implements(proc=OutputBridge, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyLoihiFixedPointOutputBridgeProcessModel(PyLoihiProcessModel):
    """PyLoihiFixedPointProcessModel for the OutputBridge Process."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._send_pipe: mp.connection.PipeConnection = self.proc_params["send_pipe"]

    def run_spk(self) -> None:
        self._send_pipe.send(self.in_port.recv())
