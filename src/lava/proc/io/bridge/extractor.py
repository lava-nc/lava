import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort, PyInPort

from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.runtime.message_infrastructure.multiprocessing import \
    MultiProcessing

from lava.proc.io.bridge.utils import ChannelConfig, ChannelSendBufferFull, \
    ChannelRecvBufferEmpty, ChannelRecvBufferNotEmpty,\
    validate_shape, validate_buffer_size, validate_channel_config, \
    send_data_blocking, send_data_non_blocking_drop, \
    recv_empty_blocking, recv_empty_non_blocking_zeros, recv_not_empty_fifo, \
    recv_not_empty_accumulate


class Extractor(AbstractProcess):
    """Process that gets data out of Lava.

    The Extractor Process exposes a method, recv_data, which enables users to
    receive data in external Python scripts from the ProcessModel of the
    Process, while the Lava Runtime is running.
    Internally, this Process builds a channel (extractor_channel, of type
    PyPyChannel).
    The src_port of the channel lives in the ProcessModel.
    The dst_port of the channel lives in the Process.

    In the ProcessModel, data is received from this Process's InPort,
    and relayed to the extractor_channel.src_port.
    When the recv_data is called from the external Python script, data is
    received from the extractor_channel.dst_port.

    For the sending part of the extractor_channel (src_port), the following
    property can be parametrized:
        (1) When the channel buffer is full, sending new data will either:
            (a) Block until free space is available (until the receiving part
            receives an item).
            (b) Not block, and the new data will not be sent.

    For the receiving part of the extractor_channel (recv_port), the following
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
        Shape of the InPort of the Process.
    buffer_size : optional, int
        Buffer size of the extractor_channel.
    extractor_channel_config : optional, ChannelConfig
        Configuration object specifying how the src_port behaves when the
        buffer is full and how the dst_port behaves when the buffer is empty
        and not empty.
    """
    def __init__(self,
                 shape: tuple[int, ...],
                 buffer_size: ty.Optional[int] = 50,
                 extractor_channel_config: ty.Optional[ChannelConfig] =
                 ChannelConfig()) -> None:
        super().__init__()

        validate_shape(shape)
        validate_buffer_size(buffer_size)
        validate_channel_config(extractor_channel_config)

        self._shape = shape

        self._extractor_channel_config = extractor_channel_config

        self._multi_processing = MultiProcessing()
        self._multi_processing.start()

        self._extractor_channel = \
            PyPyChannel(message_infrastructure=self._multi_processing,
                        src_name="src",
                        dst_name="dst",
                        shape=self._shape,
                        dtype=float,
                        size=buffer_size)
        self._extractor_channel_dst_port = self._extractor_channel.dst_port
        self._extractor_channel_dst_port.start()

        self.proc_params["shape"] = shape
        self.proc_params["extractor_channel_config"] = extractor_channel_config
        self.proc_params["extractor_channel_src_port"] = \
            self._extractor_channel.src_port

        if self._extractor_channel_config.recv_buffer_empty == \
                ChannelRecvBufferEmpty.BLOCKING:
            self._recv_empty = recv_empty_blocking
        elif self._extractor_channel_config.recv_buffer_empty == \
                ChannelRecvBufferEmpty.NON_BLOCKING_ZEROS:
            self._recv_empty = recv_empty_non_blocking_zeros

        if self._extractor_channel_config.recv_buffer_not_empty == \
                ChannelRecvBufferNotEmpty.FIFO:
            self._recv_not_empty = recv_not_empty_fifo
        elif self._extractor_channel_config.recv_buffer_not_empty == \
                ChannelRecvBufferNotEmpty.ACCUMULATE:
            self._recv_not_empty = recv_not_empty_accumulate

        self.in_port = InPort(shape=shape)

    def recv_data(self) -> np.ndarray:
        """Receive data from the ProcessModel.

        The data is received from extractor_channel.dst_port.

        Returns
        ----------
        data : np.ndarray
            Data received.
        """
        elements_in_buffer = \
            self._extractor_channel_dst_port._queue._qsize()

        if elements_in_buffer == 0:
            data = self._recv_empty(
                dst_port=self._extractor_channel_dst_port,
                zeros=np.zeros(self._shape))
        else:
            data = self._recv_not_empty(
                dst_port=self._extractor_channel_dst_port,
                zeros=np.zeros(self._shape),
                elements_in_buffer=elements_in_buffer)

        return data

    def __del__(self) -> None:
        super().__del__()

        self._multi_processing.stop()
        self._extractor_channel_dst_port.join()

@implements(proc=Extractor, protocol=LoihiProtocol)
@requires(CPU)
class PyLoihiExtractorModel(PyLoihiProcessModel):
    in_port: PyOutPort = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._shape = self.proc_params["shape"]
        self._extractor_channel_config = \
            self.proc_params["extractor_channel_config"]
        self._extractor_channel_src_port = self.proc_params[
            "extractor_channel_src_port"]
        self._extractor_channel_src_port.start()

        if self._extractor_channel_config.send_buffer_full == \
                ChannelSendBufferFull.BLOCKING:
            self._send_data = send_data_blocking
        elif self._extractor_channel_config.send_buffer_full == \
                ChannelSendBufferFull.NON_BLOCKING_DROP:
            self._send_data = send_data_non_blocking_drop

    def run_spk(self) -> None:
        self._send_data(self._extractor_channel_src_port,
                        self.in_port.recv())

    def __del__(self) -> None:
        self._extractor_channel_src_port.join()
