# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

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
                 channel_config: ty.Optional[utils.ChannelConfig] = None) -> \
            None:
        super().__init__()

        channel_config = channel_config or utils.ChannelConfig()

        utils.validate_shape(shape)
        utils.validate_buffer_size(buffer_size)
        utils.validate_channel_config(channel_config)

        self._multi_processing = MultiProcessing()
        self._multi_processing.start()

        # Stands for Process to ProcessModel
        p_to_pm = PyPyChannel(message_infrastructure=self._multi_processing,
                              src_name="src",
                              dst_name="dst",
                              shape=shape,
                              dtype=float,
                              size=buffer_size)
        self._p_to_pm_src_port = p_to_pm.src_port
        self._p_to_pm_src_port.start()

        self.proc_params["shape"] = shape
        self.proc_params["channel_config"] = channel_config
        self.proc_params["p_to_pm_dst_port"] = p_to_pm.dst_port

        self._send = channel_config.get_send_full_function()

        self.out_port = OutPort(shape=shape)

    def send(self, data: np.ndarray) -> None:
        """Send data to the ProcessModel.

        The data is sent through p_to_pm.src_port.

        Parameters
        ----------
        data : np.ndarray
            Data to be sent.
        """
        self._send(self._p_to_pm_src_port, data)

    def __del__(self) -> None:
        super().__del__()

        self._multi_processing.stop()
        self._p_to_pm_src_port.join()


@implements(proc=Injector, protocol=LoihiProtocol)
@requires(CPU)
class PyLoihiInjectorModel(PyLoihiProcessModel):
    """PyLoihiProcessModel for the Injector Process."""
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params=proc_params)

        shape = self.proc_params["shape"]
        channel_config = self.proc_params["channel_config"]
        self._p_to_pm_dst_port = self.proc_params["p_to_pm_dst_port"]
        self._p_to_pm_dst_port.start()

        self._zeros = np.zeros(shape)

        self._receive_when_empty = channel_config.get_receive_empty_function()
        self._receive_when_not_empty = \
            channel_config.get_receive_not_empty_function()

    def run_spk(self) -> None:
        self._zeros.fill(0)
        elements_in_buffer = self._p_to_pm_dst_port._queue.qsize()

        if elements_in_buffer == 0:
            data = self._receive_when_empty(
                self._p_to_pm_dst_port,
                self._zeros)
        else:
            data = self._receive_when_not_empty(
                self._p_to_pm_dst_port,
                self._zeros,
                elements_in_buffer)

        self.out_port.send(data)

    def __del__(self) -> None:
        self._p_to_pm_dst_port.join()
