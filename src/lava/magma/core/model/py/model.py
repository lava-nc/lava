# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC, abstractmethod

import numpy as np

from lava.magma.compiler.channels.pypychannel import CspSendPort, CspRecvPort, \
    CspSelector
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.py.ports import AbstractPyPort, PyVarPort
from lava.magma.runtime.mgmt_token_enums import (
    enum_to_np,
    enum_equal,
    MGMT_COMMAND,
    MGMT_RESPONSE, )


class AbstractPyProcessModel(AbstractProcessModel, ABC):
    """Abstract interface for Python ProcessModels.

    Example for how variables and ports might be initialized:
        a_in: PyInPort =   LavaPyType(PyInPort.VEC_DENSE, float)
        s_out: PyInPort =  LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
        u: np.ndarray =    LavaPyType(np.ndarray, np.int32, precision=24)
        v: np.ndarray =    LavaPyType(np.ndarray, np.int32, precision=24)
        bias: np.ndarray = LavaPyType(np.ndarray, np.int16, precision=12)
        du: int =          LavaPyType(int, np.uint16, precision=12)
    """

    def __init__(self):
        super().__init__()
        self.model_id: ty.Optional[int] = None
        self.service_to_process: ty.Optional[CspRecvPort] = None
        self.process_to_service: ty.Optional[CspSendPort] = None
        self.py_ports: ty.List[AbstractPyPort] = []
        self.var_ports: ty.List[PyVarPort] = []
        self.var_id_to_var_map: ty.Dict[int, ty.Any] = {}
        self.proc_params: ty.Dict[str, ty.Any] = {}
        self._selector: CspSelector = CspSelector()
        self._action: str = 'cmd'
        self._channel_actions: ty.List[ty.Tuple[ty.Union[CspSendPort,
                                                         CspRecvPort],
                                                ty.Callable]] = []
        self._cmd_handlers: ty.Dict[MGMT_COMMAND, ty.Callable] = {
            MGMT_COMMAND.STOP[0]: self._stop,
            MGMT_COMMAND.PAUSE[0]: self._pause,
            MGMT_COMMAND.GET_DATA[0]: self._handle_get_var,
            MGMT_COMMAND.SET_DATA[0]: self._handle_set_var
        }

    def __setattr__(self, key: str, value: ty.Any):
        self.__dict__[key] = value
        if isinstance(value, AbstractPyPort):
            self.py_ports.append(value)
            # Store all VarPorts for efficient RefPort -> VarPort handling
            if isinstance(value, PyVarPort):
                self.var_ports.append(value)

    def start(self):
        self.service_to_process.start()
        self.process_to_service.start()
        for p in self.py_ports:
            p.start()
        self._run()

    def _stop(self):
        self.process_to_service.send(MGMT_RESPONSE.TERMINATED)
        self.join()

    def _pause(self):
        self.process_to_service.send(MGMT_RESPONSE.PAUSED)

    def _handle_get_var(self):
        """Handles the get Var command from runtime service."""
        # 1. Receive Var ID and retrieve the Var
        var_id = int(self.service_to_process.recv()[0].item())
        var_name = self.var_id_to_var_map[var_id]
        var = getattr(self, var_name)

        # 2. Send Var data
        data_port = self.process_to_service
        # Header corresponds to number of values
        # Data is either send once (for int) or one by one (array)
        if isinstance(var, int) or isinstance(var, np.integer):
            data_port.send(enum_to_np(1))
            data_port.send(enum_to_np(var))
        elif isinstance(var, np.ndarray):
            # FIXME: send a whole vector (also runtime_service.py)
            var_iter = np.nditer(var)
            num_items: np.integer = np.prod(var.shape)
            data_port.send(enum_to_np(num_items))
            for value in var_iter:
                data_port.send(enum_to_np(value, np.float64))

    def _handle_set_var(self):
        """Handles the set Var command from runtime service."""
        # 1. Receive Var ID and retrieve the Var
        var_id = int(self.service_to_process.recv()[0].item())
        var_name = self.var_id_to_var_map[var_id]
        var = getattr(self, var_name)

        # 2. Receive Var data
        data_port = self.service_to_process
        if isinstance(var, int) or isinstance(var, np.integer):
            # First item is number of items (1) - not needed
            data_port.recv()
            # Data to set
            buffer = data_port.recv()[0]
            if isinstance(var, int):
                setattr(self, var_name, buffer.item())
            else:
                setattr(self, var_name, buffer.astype(var.dtype))
        elif isinstance(var, np.ndarray):
            # First item is number of items
            num_items = data_port.recv()[0]
            var_iter = np.nditer(var, op_flags=['readwrite'])
            # Set data one by one
            for i in var_iter:
                if num_items == 0:
                    break
                num_items -= 1
                i[...] = data_port.recv()[0]
        else:
            raise RuntimeError("Unsupported type")

    def _handle_var_port(self, var_port):
        """Handles read/write requests on the given VarPort."""
        var_port.service()

    def _run(self):
        while True:
            if self._action == 'cmd':
                cmd = self.service_to_process.recv()[0]
                try:
                    if cmd in self._cmd_handlers:
                        self._cmd_handlers[cmd]()
                        if cmd == MGMT_COMMAND.STOP[0]:
                            break
                    else:
                        raise ValueError(
                            f"Illegal RuntimeService command! ProcessModels of "
                            f"type {self.__class__.__qualname__} cannot handle "
                            f"command: {cmd}")
                except Exception as inst:
                    # Inform runtime service about termination
                    self.process_to_service.send(MGMT_RESPONSE.ERROR)
                    self.join()
                    raise inst
            else:
                # Handle VarPort requests from RefPorts
                self._handle_var_port(self._action)
            self._channel_actions = [(self.service_to_process, lambda: 'cmd')]
            self.run()
            self._action = self._selector.select(*self._channel_actions)

    @abstractmethod
    def run(self):
        pass

    def join(self):
        self.service_to_process.join()
        self.process_to_service.join()
        for p in self.py_ports:
            p.join()


class PyLoihiProcessModel(AbstractPyProcessModel):
    def __init__(self):
        super(PyLoihiProcessModel, self).__init__()
        self.current_ts = 0
        self.phase = PyLoihiProcessModel.Phase.SPK
        self._cmd_handlers.update({
            PyLoihiProcessModel.Phase.SPK[0]: self._spike,
            PyLoihiProcessModel.Phase.PRE_MGMT[0]: self._pre_mgmt,
            PyLoihiProcessModel.Phase.LRN[0]: self._lrn,
            PyLoihiProcessModel.Phase.POST_MGMT[0]: self._post_mgmt,
            PyLoihiProcessModel.Phase.HOST[0]: self._host
        })
        pass

    class Phase:
        SPK = enum_to_np(1)
        PRE_MGMT = enum_to_np(2)
        LRN = enum_to_np(3)
        POST_MGMT = enum_to_np(4)
        HOST = enum_to_np(5)

    def run_spk(self):
        pass

    def run_pre_mgmt(self):
        pass

    def run_lrn(self):
        pass

    def run_post_mgmt(self):
        pass

    def pre_guard(self):
        pass

    def lrn_guard(self):
        pass

    def post_guard(self):
        pass

    def _spike(self):
        self.current_ts += 1
        self.phase = PyLoihiProcessModel.Phase.SPK
        self.run_spk()
        self.process_to_service.send(MGMT_RESPONSE.DONE)

    def _pre_mgmt(self):
        self.phase = PyLoihiProcessModel.Phase.PRE_MGMT
        if self.pre_guard():
            self.run_pre_mgmt()
        self.process_to_service.send(MGMT_RESPONSE.DONE)

    def _post_mgmt(self):
        self.phase = PyLoihiProcessModel.Phase.POST_MGMT
        if self.post_guard():
            self.run_post_mgmt()
        self.process_to_service.send(MGMT_RESPONSE.DONE)

    def _lrn(self):
        self.phase = PyLoihiProcessModel.Phase.LRN
        if self.lrn_guard():
            self.run_lrn()
        self.process_to_service.send(MGMT_RESPONSE.DONE)

    def _host(self):
        self.phase = PyLoihiProcessModel.Phase.HOST

    def run(self):
        """Retrieves commands from the runtime service to iterate through the
        phases of Loihi and calls their corresponding methods of the
        ProcessModels. The phase is retrieved from runtime service
        (service_to_process). After calling the method of a phase of all
        ProcessModels the runtime service is informed about completion. The
        loop ends when the STOP command is received."""
        if enum_equal(self.phase, PyLoihiProcessModel.Phase.PRE_MGMT) or \
                enum_equal(self.phase, PyLoihiProcessModel.Phase.POST_MGMT):
            for var_port in self.var_ports:
                for csp_port in var_port.csp_ports:
                    if isinstance(csp_port, CspRecvPort):
                        self._channel_actions.append((csp_port, lambda:
                        var_port))


