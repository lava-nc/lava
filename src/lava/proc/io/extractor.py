from collections import deque

import numpy as np
import typing as ty
from enum import IntEnum, auto

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort, InPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort, PyInPort

from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.runtime.message_infrastructure.multiprocessing import \
    MultiProcessing

#For now there is actually only one mode we can do
#class ExtractorMode(IntEnum):
#    FIFO_BATCH = auto()
#    LIFO_CLEAR = auto()
#    ACCUMULATE = auto()

class Extractor(AbstractProcess):
    def __init__(self,
                 shape: tuple[int, ...],
                 dtype: ty.Union[ty.Type, np.dtype],
                 size: int) -> None:
        super().__init__()

        self._validate_shape(shape)
        self._validate_dtype(dtype)
        self._validate_size(size)
       # self._validate_extractor_mode(extractor_mode)
        self._shape = shape

        self._multi_processing = MultiProcessing()
        self._multi_processing.start()

        self._extractor_channel = \
            PyPyChannel(message_infrastructure=self._multi_processing,
                        src_name="src",
                        dst_name="dst",
                        shape=self._shape,
                        dtype=dtype,
                        size=size)
        self._extractor_channel_dst_port = self._extractor_channel.dst_port
        self._extractor_channel_dst_port.start()

        self.proc_params["shape"] = shape
        self.proc_params["extractor_channel_src_port"] = \
            self._extractor_channel.src_port
        #self.proc_params["extractor_mode"] =extractor_mode

        self.in_port = InPort(shape=shape)

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

    #@staticmethod
    #def _validate_extractor_mode(extractor_mode: ExtractorMode) ->\
    #        None:
    #    if not isinstance(extractor_mode, ExtractorMode):
    #        raise TypeError("Expected <extractor_mode> to be of type "
    #                        "ExtractorMode. Got <extractor_mode> = "
    #                        f"{extractor_mode}.")

    def rcv_data(self) -> np.ndarray:
        data = np.zeros(shape=self._shape)
        if self._extractor_channel_dst_port.probe():
            data += self._extractor_channel_dst_port.recv()
        return data

    def __del__(self) -> None:
        super().__del__()

        self._multi_processing.stop()
        self._extractor_channel_dst_port.join()


class PyExtractorModel(PyLoihiProcessModel):
    in_port = None

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        self._shape = self.proc_params["shape"]
        self._extractor_channel_src_port = self.proc_params[
            "extractor_channel_src_port"]
        #self._extractor_mode = self.proc_params["extractor_mode"]
        self._extractor_channel_src_port.start()

    def run_spk(self) -> None:
        data = self.in_port.recv()
        if self._extractor_channel_src_port.probe():
            print(data)
            self._extractor_channel_src_port.send(data)

    def __del__(self) -> None:
        self._extractor_channel_src_port.join()

@implements(proc=Extractor, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyExtractorModelFloat(PyExtractorModel):
    in_port: PyOutPort = LavaPyType(PyInPort.VEC_DENSE, float)

@implements(proc=Extractor, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyExtractorModelFixed(PyExtractorModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)


