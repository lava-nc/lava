# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort, Var
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.model import PyAsyncProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.runtime.message_infrastructure.multiprocessing import \
    MultiProcessing
from lava.proc.io import utils


class Extractor(AbstractProcess):
    """Extractor allows non-Lava code, such as a third-party Python library
    to extract data from a Lava Process while the Lava Runtime is running,
    by calling receive.

    Internally, this Process builds a channel from the ProcessModel to the
    Process (named pm_to_p, of type PyPyChannel).
    The src_port of the channel lives in the ProcessModel.
    The dst_port of the channel lives in the Process.

    In the ProcessModel, data is received from this Process's InPort, and
    relayed to the pm_to_p.src_port.
    When the receive method is called from the external Python script, data is
    received from the pm_to_p.dst_port.

    Parameters
    ----------
    shape : tuple
        Shape of the InPort of the Process, and of the np.ndarrays passed
        through the channel between the ProcessModel and the Process.
    buffer_size : int, optional
        Buffer size (in terms of number of np.ndarrays) of the channel between
        the ProcessModel and Process.
    channel_config : ChannelConfig, optional
        Configuration object specifying how the src_port behaves when the
        buffer is full and how the dst_port behaves when the buffer is empty
        and not empty.
    """

    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 buffer_size: ty.Optional[int] = 50,
                 channel_config: ty.Optional[utils.ChannelConfig] = None,
                 **kwargs) -> None:
        super().__init__(shape_1=shape, **kwargs)

        channel_config = channel_config or utils.ChannelConfig()

        utils.validate_shape(shape)
        utils.validate_buffer_size(buffer_size)
        utils.validate_channel_config(channel_config)

        self._shape = shape

        self.proc_params["channel_config"] = channel_config

        self._receive_when_empty = channel_config.get_receive_empty_function()
        self._receive_when_not_empty = \
            channel_config.get_receive_not_empty_function()

        self.in_port = InPort(shape=shape)
        self.out_port = OutPort(shape=shape)
        self.out_port.flag_external_pipe(buffer_size=buffer_size)

    def receive(self) -> np.ndarray:
        """Receive data from the ProcessModel.

        The data is received from out_port.

        Returns
        ----------
        data : np.ndarray
            Data received.
        """
        if not hasattr(self.out_port, 'external_pipe_csp_recv_port'):
            raise AssertionError("The Runtime needs to be created before"
                                 "calling <send>. Please use the method "
                                 "<create_runtime> or <run> on your Lava"
                                 " network before using <send>.")

        elements_in_buffer = \
            self.out_port.external_pipe_csp_recv_port._queue.qsize()

        if elements_in_buffer == 0:
            data = self._receive_when_empty(
                self.out_port.external_pipe_csp_recv_port,
                np.zeros(self._shape))
        else:
            data = self._receive_when_not_empty(
                self.out_port.external_pipe_csp_recv_port,
                np.zeros(self._shape),
                elements_in_buffer)

        return data


@implements(proc=Extractor, protocol=LoihiProtocol)
@requires(CPU)
class PyLoihiExtractorModel(PyLoihiProcessModel):
    """PyLoihiProcessModel for the Extractor Process."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        channel_config = self.proc_params["channel_config"]

        self._send = channel_config.get_send_full_function()

    def run_spk(self) -> None:
        self._send(self.out_port.csp_ports[-1],
                   self.in_port.recv())


@implements(proc=Extractor, protocol=AsyncProtocol)
@requires(CPU)
class PyLoihiExtractorModelAsync(PyAsyncProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        channel_config = self.proc_params["channel_config"]

        self._send = channel_config.get_send_full_function()
        self.time_step = 1

    def run_async(self) -> None:
        while self.time_step != self.num_steps + 1:
            self._send(self.out_port.csp_ports[-1],
                       self.in_port.recv())
            self.time_step += 1


class VarWire(AbstractProcess):
    """VarWire allows non-Lava code, such as a third-party Python library
    to tap data from a Lava Process Variable (Var) while the Lava Runtime is
    running, by calling receive.

    Internally, this Process builds a channel from the ProcessModel to the
    Process (named pm_to_p, of type PyPyChannel).
    The wire_tap (ref_port) of the channel lives in the ProcessModel.
    The dst_port of the channel lives in the Process.

    In the ProcessModel, data is received from this Process's RefPort, and
    relayed to the pm_to_p.src_port.
    When the receive method is called from the external Python script, data is
    received from the pm_to_p.dst_port.

    Parameters
    ----------
    shape : tuple
        Shape of the InPort of the Process, and of the np.ndarrays passed
        through the channel between the ProcessModel and the Process.
    buffer_size : int, optional
        Buffer size (in terms of number of np.ndarrays) of the channel between
        the ProcessModel and Process.
    channel_config : ChannelConfig, optional
        Configuration object specifying how the src_port behaves when the
        buffer is full and how the dst_port behaves when the buffer is empty
        and not empty.
    """

    def __init__(self,
                 buffer_size: ty.Optional[int] = 50,
                 channel_config: ty.Optional[utils.ChannelConfig] = None) -> \
            None:
        super().__init__()

        channel_config = channel_config or utils.ChannelConfig()

        utils.validate_buffer_size(buffer_size)
        utils.validate_channel_config(channel_config)

        self.shape = None
        self.buffer_size = buffer_size
        self._multi_processing = None
        # Stands for ProcessModel to Process
        self._pm_to_p_dst_port = None

        self.proc_params["channel_config"] = channel_config

        self._receive_when_empty = channel_config.get_receive_empty_function()
        self._receive_when_not_empty = \
            channel_config.get_receive_not_empty_function()

        self.wire_tap = RefPort((1,))

    def connect_var(self, var: Var) -> None:
        self.shape = var.shape
        self.wire_tap = RefPort(self.shape)
        self.wire_tap.connect_var(var)

        self._multi_processing = MultiProcessing()
        self._multi_processing.start()

        # Stands for ProcessModel to Process
        pm_to_p = PyPyChannel(message_infrastructure=self._multi_processing,
                              src_name="src",
                              dst_name="dst",
                              shape=self.shape,
                              dtype=float,
                              size=self.buffer_size)
        self._pm_to_p_dst_port = pm_to_p.dst_port
        self._pm_to_p_dst_port.start()

        self.proc_params["pm_to_p_src_port"] = pm_to_p.src_port
        self._post_init()

    def receive(self) -> np.ndarray:
        """Receive data from the ProcessModel.

        The data is received from pm_to_p.dst_port.

        Returns
        ----------
        data : np.ndarray
            Data received.
        """
        elements_in_buffer = self._pm_to_p_dst_port._queue.qsize()

        if elements_in_buffer == 0:
            data = self._receive_when_empty(
                self._pm_to_p_dst_port,
                np.zeros(self.shape))
        else:
            data = self._receive_when_not_empty(
                self._pm_to_p_dst_port,
                np.zeros(self.shape),
                elements_in_buffer)

        return data

    def __del__(self) -> None:
        super().__del__()

        self._multi_processing.stop()
        self._pm_to_p_dst_port.join()


@implements(proc=VarWire, protocol=LoihiProtocol)
@requires(CPU)
class PyLoihiVarWireModel(PyLoihiProcessModel):
    wire_tap: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        channel_config = self.proc_params["channel_config"]
        self._pm_to_p_src_port = self.proc_params["pm_to_p_src_port"]
        self._pm_to_p_src_port.start()

        self._send = channel_config.get_send_full_function()

    def post_guard(self) -> None:
        return True

    def run_post_mgmt(self) -> None:
        self._send(self._pm_to_p_src_port, self.wire_tap.read())

    def __del__(self) -> None:
        self._pm_to_p_src_port.join()
