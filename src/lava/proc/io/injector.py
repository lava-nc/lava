import numpy as np
import typing as ty
from enum import IntEnum, auto

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort

from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.runtime.message_infrastructure.multiprocessing import \
    MultiProcessing


class InjectorSendMode(IntEnum):
    BLOCKING = auto()
    NON_BLOCKING_ERROR = auto()
    NON_BLOCKING_CIRCULAR_BUFFER = auto()


class InjectorRecvMode(IntEnum):
    FIFO = auto()
    LIFO_CLEAR = auto()
    ACCUMULATE = auto()


class Injector(AbstractProcess):
    def __init__(self,
                 shape: tuple[int, ...],
                 dtype: ty.Union[ty.Type, np.dtype],
                 size: int,
                 injector_send_mode: ty.Optional[InjectorSendMode] =
                 InjectorSendMode.BLOCKING,
                 injector_recv_mode: ty.Optional[InjectorRecvMode] =
                 InjectorRecvMode.FIFO) -> None:
        super().__init__()

        self._validate_shape(shape)
        self._validate_dtype(dtype)
        self._validate_size(size)
        self._validate_injector_send_mode(injector_send_mode)
        self._validate_injector_recv_mode(injector_recv_mode)

        self._injector_send_mode = injector_send_mode

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
        self.proc_params["injector_recv_mode"] = injector_recv_mode
        self.proc_params["injector_channel_dst_port"] = \
            self._injector_channel.dst_port

        self.out_port = OutPort(shape=shape)

    @staticmethod
    def _validate_shape(shape):
        if not isinstance(shape, tuple):
            raise TypeError("Expected <shape> to be of type tuple. Got "
                            f"<shape> = {shape}.")

        for s in shape:
            if not isinstance(s, int):
                raise TypeError("Expected all elements of <shape> to be of "
                                f"type int. Got <shape> = {shape}.")
            if s <= 0:
                raise ValueError("Expected all elements of <shape> to be "
                                 f"strictly positive. Got <shape> = {shape}.")

    @staticmethod
    def _validate_dtype(dtype: ty.Union[ty.Type, np.dtype]) -> None:
        if not isinstance(dtype, (type, np.dtype)):
            raise TypeError("Expected <dtype> to be of type type or np.dtype. "
                            f"Got <dtype> = {dtype}.")

    @staticmethod
    def _validate_size(size):
        if not isinstance(size, int):
            raise TypeError("Expected <size> to be of type int. Got <size> = "
                            f"{size}.")
        if size <= 0:
            raise ValueError("Expected <size> to be strictly positive. Got "
                             f"<size> = {size}.")

    @staticmethod
    def _validate_injector_send_mode(injector_send_mode: InjectorSendMode) ->\
            None:
        if not isinstance(injector_send_mode, InjectorSendMode):
            raise TypeError("Expected <injector_send_mode> to be of type "
                            "InjectorSendMode. Got <injector_send_mode> = "
                            f"{injector_send_mode}.")

        if injector_send_mode == InjectorSendMode.NON_BLOCKING_ERROR or \
                injector_send_mode == \
                InjectorSendMode.NON_BLOCKING_CIRCULAR_BUFFER:
            raise NotImplementedError()

    @staticmethod
    def _validate_injector_recv_mode(injector_recv_mode: InjectorRecvMode) ->\
            None:
        if not isinstance(injector_recv_mode, InjectorRecvMode):
            raise TypeError("Expected <injector_recv_mode> to be of type "
                            "InjectorRecvMode. Got <injector_recv_mode> = "
                            f"{injector_recv_mode}.")

        if injector_recv_mode == InjectorRecvMode.LIFO_CLEAR:
            raise NotImplementedError()

    def send_data(self, data: np.ndarray) -> None:
        self._validate_data(data)
        # self._validate_runtime()

        self._injector_channel_src_port.send(data)

    def _validate_data(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("Expected <data> to be of type np.ndarray. Got "
                            f"<data> = {data}")

        if data.shape != self.out_port.shape:
            raise ValueError("Expected <data>.shape to be equal to shape of "
                             f"OutPort. Got <data>.shape = {data.shape} and "
                             f"<out_port>.shape = {self.out_port.shape}.")

    # def _validate_runtime(self) -> None:
    #     if self.runtime is None or not self.runtime._is_started:
    #         raise RuntimeError("Data can only be sent when a Runtime has been "
    #                            "started.")

    def __del__(self) -> None:
        super().__del__()

        self._multi_processing.stop()
        self._injector_channel_src_port.join()


class PyInjectorModel(PyLoihiProcessModel):
    out_port = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._shape = self.proc_params["shape"]
        self._injector_recv_mode = self.proc_params["injector_recv_mode"]
        self._injector_channel_dst_port = self.proc_params[
            "injector_channel_dst_port"]

        self._injector_channel_dst_port.start()

        if self._injector_recv_mode == InjectorRecvMode.FIFO:
            self.run_spk = self.run_spk_fifo
        elif self._injector_recv_mode == InjectorRecvMode.LIFO_CLEAR:
            self.run_spk = self.run_spk_lifo_clear
        elif self._injector_recv_mode == InjectorRecvMode.ACCUMULATE:
            self.run_spk = self.run_spk_accumulate

    def run_spk_fifo(self) -> None:
        data = np.zeros(self._shape)

        if self._injector_channel_dst_port.probe():
            data = self._injector_channel_dst_port.recv()

        self.out_port.send(data)

    def run_spk_lifo_clear(self) -> None:
        raise NotImplementedError()

    def run_spk_accumulate(self) -> None:
        data = np.zeros(self._shape)

        elements_in_queue = self._injector_channel_dst_port._queue._qsize()
        for _ in range(elements_in_queue):
            data += self._injector_channel_dst_port.recv()

        self.out_port.send(data)

    def __del__(self) -> None:
        self._injector_channel_dst_port.join()


@implements(proc=Injector, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyInjectorModelFloat(PyInjectorModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)


@implements(proc=Injector, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyInjectorModelFixed(PyInjectorModel):
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
