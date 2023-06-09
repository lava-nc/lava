from abc import abstractmethod

import numpy as np
import typing as ty
import warnings

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort

from lava.magma.runtime.message_infrastructure.multiprocessing import \
    MultiProcessing
from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.proc.io.bridge.utils import ChannelConfig, ChannelSendBufferFull, \
    ChannelRecvBufferEmpty, ChannelRecvBufferNotEmpty,\
    validate_shape, validate_dtype, validate_size, validate_channel_config, \
    validate_send_data, send_data_blocking, send_data_non_blocking_drop, \
    recv_empty_blocking, recv_empty_non_blocking_zeros, recv_not_empty_fifo, \
    recv_not_empty_accumulate


class Injector(AbstractProcess):
    def __init__(self,
                 shape: tuple[int, ...],
                 dtype: ty.Union[ty.Type, np.dtype],
                 size: int,
                 injector_channel_config: ty.Optional[ChannelConfig] =
                 ChannelConfig()) \
            -> None:
        super().__init__()

        validate_shape(shape)
        validate_dtype(dtype)
        validate_size(size)
        validate_channel_config(injector_channel_config)

        self._injector_channel_config = injector_channel_config

        self._multi_processing = MultiProcessing()
        self._multi_processing.start()

        self._injector_channel = \
            PyPyChannel(message_infrastructure=self._multi_processing,
                        src_name="src",
                        dst_name="dst",
                        shape=shape,
                        dtype=dtype,
                        size=size)
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
        validate_send_data(data, self.out_port.shape)
        self._send_data(self._injector_channel_src_port, data)

    def __del__(self) -> None:
        super().__del__()

        self._multi_processing.stop()
        self._injector_channel_src_port.join()


class PyInjectorModel(PyLoihiProcessModel):
    out_port = None

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

    @abstractmethod
    def _zeros(self) -> np.ndarray:
        pass

    def run_spk(self) -> None:
        elements_in_channel_queue = \
            self._injector_channel_dst_port._queue._qsize()

        if elements_in_channel_queue == 0:
            data = self._recv_empty(
                dst_port=self._injector_channel_dst_port,
                zeros=self._zeros())
        else:
            data = self._recv_not_empty(
                dst_port=self._injector_channel_dst_port,
                zeros=self._zeros(),
                elements_in_queue=elements_in_channel_queue)

        self.out_port.send(data)


@implements(proc=Injector, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyInjectorModelFloat(PyInjectorModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def _zeros(self) -> np.ndarray:
        return np.zeros(self._shape, dtype=float)


@implements(proc=Injector, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyInjectorModelFixed(PyInjectorModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def _zeros(self) -> np.ndarray:
        return np.zeros(self._shape, dtype=int)
