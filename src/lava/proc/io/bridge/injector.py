import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort

from lava.magma.runtime.message_infrastructure.multiprocessing import \
    MultiProcessing
from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.proc.io.bridge.utils import ChannelConfig, ChannelSendBufferFull, \
    ChannelRecvBufferEmpty, ChannelRecvBufferNotEmpty,\
    validate_shape, validate_buffer_size, validate_channel_config, \
    validate_send_data, send_data_blocking, send_data_non_blocking_drop, \
    recv_empty_blocking, recv_empty_non_blocking_zeros, recv_not_empty_fifo, \
    recv_not_empty_accumulate


class Injector(AbstractProcess):
    """Process that gets data into Lava.

    The Injector Process exposes a method, send_data, which enables users to
    send data from external Python scripts to the ProcessModel of the
    Process, while the Lava Runtime is running.
    Internally, this Process builds a channel (injector_channel, of type
    PyPyChannel).
    The src_port of the channel lives in the Process.
    The dst_port of the channel lives in the ProcessModel.

    When the send_data method is called from the external Python script, data is
    sent through the injector_channel.src_port.
    In the ProcessModel, data is received through the
    injector_channel.recv_port, and relayed to this Process's OutPort.

    For the sending part of the injector_channel (src_port), the following
    property can be parametrized:
        (1) When the channel buffer is full, sending new data will either:
            (a) Block until free space is available (until the receiving part
            receives an item).
            (b) Not block, and the new data will not be sent.

    For the receiving part of the injector_channel (recv_port), the following
    properties can be parametrized:
        (1) When the channel buffer is empty, receiving will either:
            (a) Block until an item is available (until the sending part
            sends an item).
            (b) Not block, and the received data will be zeros.
        (2) When the channel buffer is not empty, receiving will either:
            (a) Receive a single item, being the oldest item put in the
            channel (FIFO).
            (b) Receive all items available, accumulated.

    Parameters
    ----------
    shape : tuple
        Shape of the OutPort of the Process.
    buffer_size : optional, int
        Buffer size of the injector_channel.
    injector_channel_config : optional, ChannelConfig
        Configuration object specifying how the src_port behaves when the
        buffer is full and how the dst_port behaves when the buffer is empty
        and not empty.
    """
    def __init__(self,
                 shape: tuple[int, ...],
                 buffer_size: ty.Optional[int] = 50,
                 injector_channel_config: ty.Optional[ChannelConfig] =
                 ChannelConfig()) \
            -> None:
        super().__init__()

        validate_shape(shape)
        validate_buffer_size(buffer_size)
        validate_channel_config(injector_channel_config)

        self._injector_channel_config = injector_channel_config

        self._multi_processing = MultiProcessing()
        self._multi_processing.start()

        self._injector_channel = \
            PyPyChannel(message_infrastructure=self._multi_processing,
                        src_name="src",
                        dst_name="dst",
                        shape=shape,
                        dtype=float,
                        size=buffer_size)
        self._injector_channel_src_port = self._injector_channel.src_port
        self._injector_channel_src_port.start()

        self.proc_params["shape"] = shape
        self.proc_params["injector_channel_config"] = injector_channel_config
        self.proc_params["injector_channel_dst_port"] = \
            self._injector_channel.dst_port

        if self._injector_channel_config.send_buffer_full == \
                ChannelSendBufferFull.BLOCKING:
            self._send_data = send_data_blocking
        elif self._injector_channel_config.send_buffer_full == \
                ChannelSendBufferFull.NON_BLOCKING_DROP:
            self._send_data = send_data_non_blocking_drop

        self.out_port = OutPort(shape=shape)

    def send_data(self, data: np.ndarray) -> None:
        """Send data to the ProcessModel.

        The data is sent through injector_channel.src_port.

        Parameters
        ----------
        data : np.ndarray
            Data to be sent.
        """
        validate_send_data(data, self.out_port)
        self._send_data(self._injector_channel_src_port, data)

    def __del__(self) -> None:
        super().__del__()

        self._multi_processing.stop()
        self._injector_channel_src_port.join()


@implements(proc=Injector, protocol=LoihiProtocol)
@requires(CPU)
class PyLoihiInjectorModel(PyLoihiProcessModel):
    """PyLoihiProcessModel for the Injector Process."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._shape = self.proc_params["shape"]
        self._injector_channel_config = \
            self.proc_params["injector_channel_config"]
        self._injector_channel_dst_port = \
            self.proc_params["injector_channel_dst_port"]

        self._injector_channel_dst_port.start()

        if self._injector_channel_config.recv_buffer_empty == \
                ChannelRecvBufferEmpty.BLOCKING:
            self._recv_empty = recv_empty_blocking
        elif self._injector_channel_config.recv_buffer_empty == \
                ChannelRecvBufferEmpty.NON_BLOCKING_ZEROS:
            self._recv_empty = recv_empty_non_blocking_zeros

        if self._injector_channel_config.recv_buffer_not_empty == \
                ChannelRecvBufferNotEmpty.FIFO:
            self._recv_not_empty = recv_not_empty_fifo
        elif self._injector_channel_config.recv_buffer_not_empty == \
                ChannelRecvBufferNotEmpty.ACCUMULATE:
            self._recv_not_empty = recv_not_empty_accumulate

    def run_spk(self) -> None:
        elements_in_buffer = \
            self._injector_channel_dst_port._queue._qsize()

        if elements_in_buffer == 0:
            data = self._recv_empty(
                dst_port=self._injector_channel_dst_port,
                zeros=np.zeros(self._shape))
        else:
            data = self._recv_not_empty(
                dst_port=self._injector_channel_dst_port,
                zeros=np.zeros(self._shape),
                elements_in_buffer=elements_in_buffer)

        self.out_port.send(data)
