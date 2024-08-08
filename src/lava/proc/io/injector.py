# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.model import PyAsyncProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.proc.io import utils


class Injector(AbstractProcess):
    """Injector allows non-Lava code, such as a third-party Python library
    to inject data to a Lava Process while the Lava Runtime is running,
    by calling send.

    Internally, this Process builds a channel from the Process to the
    ProcessModel (named p_to_pm, of type PyPyChannel).
    The src_port of the channel lives in the Process.
    The dst_port of the channel lives in the ProcessModel.

    When the send method is called from the external Python script, data is
    sent through the p_to_pm.src_port.
    In the ProcessModel, data is received through the p_to_pm.dst_port,
    and relayed to this Process's OutPort.

    Parameters
    ----------
    shape : tuple
        Shape of the OutPort of the Process, and of the np.ndarrays passed
        through the channel between the Process and the ProcessModel.
    buffer_size : int, optional
        Buffer size (in terms of number of np.ndarrays) of the channel between
        the Process and ProcessModel.
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

        self.in_port = InPort(shape=shape)
        self.in_port.flag_external_pipe(buffer_size=buffer_size)
        self.out_port = OutPort(shape=shape)

        self.proc_params["shape"] = shape
        self.proc_params["channel_config"] = channel_config

        self._send = channel_config.get_send_full_function()

    def send(self, data: np.ndarray) -> None:
        """Send data to connected process.

        Parameters
        ----------
        data : np.ndarray
            Data to be sent.

        Raises
        ------
        AssertionError
            If the runtime of the Lava network was not created.
        """
        # The csp channel is created by the runtime
        if hasattr(self.in_port, 'external_pipe_csp_send_port'):
            self._send(self.in_port.external_pipe_csp_send_port, data)
        else:
            raise AssertionError("The Runtime needs to be created before"
                                 "calling <send>. Please use the method "
                                 "<create_runtime> or <run> on your Lava"
                                 " network before using <send>.")


@implements(proc=Injector, protocol=LoihiProtocol)
@requires(CPU)
class PyLoihiInjectorModel(PyLoihiProcessModel):
    """PyLoihiProcessModel for the Injector Process."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        shape = self.proc_params["shape"]
        channel_config = self.proc_params["channel_config"]
        self._zeros = np.zeros(shape)

        self._receive_when_empty = channel_config.get_receive_empty_function()
        self._receive_when_not_empty = \
            channel_config.get_receive_not_empty_function()

    def run_spk(self) -> None:
        self._zeros.fill(0)
        elements_in_buffer = self.in_port.csp_ports[-1]._queue.qsize()

        if elements_in_buffer == 0:
            data = self._receive_when_empty(
                self.in_port,
                self._zeros)
        else:
            data = self._receive_when_not_empty(
                self.in_port,
                self._zeros,
                elements_in_buffer)

        self.out_port.send(data)

    def __del__(self) -> None:
        self._p_to_pm_dst_port.join()


@implements(proc=Injector, protocol=AsyncProtocol)
@requires(CPU)
class PyLoihiInjectorModelAsync(PyAsyncProcessModel):
    """PyAsyncProcessModel for the Injector Process."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        shape = self.proc_params["shape"]
        channel_config = self.proc_params["channel_config"]
        self._zeros = np.zeros(shape)

        self._receive_when_empty = channel_config.get_receive_empty_function()
        self._receive_when_not_empty = \
            channel_config.get_receive_not_empty_function()
        self.time_step = 1

    def run_async(self) -> None:
        while self.time_step != self.num_steps + 1:
            self._zeros.fill(0)
            elements_in_buffer = self.in_port.csp_ports[-1]._queue.qsize()

            if elements_in_buffer == 0:
                data = self._receive_when_empty(
                    self.in_port,
                    self._zeros)
            else:
                data = self._receive_when_not_empty(
                    self.in_port,
                    self._zeros,
                    elements_in_buffer)

            self.out_port.send(data)
            self.time_step += 1

    def __del__(self) -> None:
        self._p_to_pm_dst_port.join()
