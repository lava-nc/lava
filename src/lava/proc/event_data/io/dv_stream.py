import typing as ty

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from dv import NetworkNumpyEventPacketInput

class DvStream(AbstractProcess):
    """
    Parameters
    ----------

    """
    def __init__(self,
                 *,
                 address: str,
                 port: int,
                 shape_out: ty.Tuple[int]) -> None:
        super().__init__(address=address,
                         port=port,
                         shape_out=shape_out)
        self._validate_shape_out(shape_out)
        self._validate_port(port)
        self._validate_address(address)
        self.out_port = OutPort(shape=shape_out)


    @staticmethod
    def _validate_shape_out(shape_out: ty.Tuple[int]) -> None:
        """
        Checks whether the given shape is valid and that the size given
        is not a negative number. Raises relevant exception if not
        """
        if len(shape_out) != 1:
            raise ValueError(f"Shape of the OutPort should be (n,). "
                             f"{shape_out} was given.")
        if shape_out[0] <= 0:
            raise ValueError(f"Max number of events should be positive. "
                             f"{shape_out} was given.")

    @staticmethod
    def _validate_port(port: int) -> None:
        """
        Check whether the given port is valid. Raises relevant exception if not
        """

        if not (0 <= port <= 65535):
            raise ValueError(f"Port should be between 0 and 65535"
                             f"{port} was given.")

    @staticmethod
    def _validate_address(address: str) -> None:
        """
        Check that address is not an ampty string. Raises relevant exception if not
        """

        if not address:
            raise ValueError("Address should not be empty")


@implements(proc=DvStream, protocol=LoihiProtocol)
@requires(CPU)
class DvStreamPM(PyLoihiProcessModel):
    """
    Implementation of the DvStream process on Loihi, with sparse
    representation of events.
    """
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self._address = proc_params["address"]
        self._port = proc_params["port"]
        self._shape_out = proc_params["shape_out"]
        self._event_stream = proc_params.get("_event_stream")
        if not self._event_stream:
            self._event_stream = NetworkNumpyEventPacketInput(address=self._address, port=self._port)


