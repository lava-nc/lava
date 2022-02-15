# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC, abstractmethod

import logging

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
from lava.magma.runtime.runtime_services.enums import LoihiPhase


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

    def __init__(self, proc_params: ty.Dict[str, ty.Any],
                 loglevel=logging.WARNING) -> None:
        super().__init__(proc_params, loglevel=loglevel)
        self.model_id: ty.Optional[int] = None
        self.service_to_process: ty.Optional[CspRecvPort] = None
        self.process_to_service: ty.Optional[CspSendPort] = None
        self.py_ports: ty.List[AbstractPyPort] = []
        self.var_ports: ty.List[PyVarPort] = []
        self.var_id_to_var_map: ty.Dict[int, ty.Any] = {}

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
        self.run()

    @abstractmethod
    def run(self):
        pass

    def join(self):
        self.service_to_process.join()
        self.process_to_service.join()
        for p in self.py_ports:
            p.join()


class PyLoihiProcessModel(AbstractPyProcessModel):
    def __init__(self, proc_params: ty.Dict[str, ty.Any],
                 loglevel=logging.WARNING):
        super(PyLoihiProcessModel, self).__init__(proc_params,
                                                  loglevel=loglevel)
        self.current_ts = 0

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

    # TODO: (PP) need to handle PAUSE command
    def run(self):
        """Retrieves commands from the runtime service to iterate through the
        phases of Loihi and calls their corresponding methods of the
        ProcessModels. The phase is retrieved from runtime service
        (service_to_process). After calling the method of a phase of all
        ProcessModels the runtime service is informed about completion. The
        loop ends when the STOP command is received."""
        selector = CspSelector()
        action = 'cmd'
        phase = LoihiPhase.SPK
        while True:
            if action == 'cmd':
                cmd = self.service_to_process.recv()
                if enum_equal(cmd, MGMT_COMMAND.STOP):
                    self.process_to_service.send(MGMT_RESPONSE.TERMINATED)
                    self.join()
                    return
                try:
                    # Spiking phase - increase time step
                    if enum_equal(cmd, LoihiPhase.SPK):
                        self.current_ts += 1
                        phase = LoihiPhase.SPK
                        self.run_spk()
                        self.process_to_service.send(MGMT_RESPONSE.DONE)
                    # Pre-management phase
                    elif enum_equal(cmd,
                                    LoihiPhase.PRE_MGMT):
                        # Enable via guard method
                        phase = LoihiPhase.PRE_MGMT
                        if self.pre_guard():
                            self.run_pre_mgmt()
                        self.process_to_service.send(MGMT_RESPONSE.DONE)
                    # Learning phase
                    elif enum_equal(cmd, LoihiPhase.LRN):
                        # Enable via guard method
                        phase = LoihiPhase.LRN
                        if self.lrn_guard():
                            self.run_lrn()
                        self.process_to_service.send(MGMT_RESPONSE.DONE)
                    # Post-management phase
                    elif enum_equal(cmd,
                                    LoihiPhase.POST_MGMT):
                        # Enable via guard method
                        phase = LoihiPhase.POST_MGMT
                        if self.post_guard():
                            self.run_post_mgmt()
                        self.process_to_service.send(MGMT_RESPONSE.DONE)
                    # Host phase - called at the last time step before STOP
                    elif enum_equal(cmd, LoihiPhase.HOST):
                        phase = LoihiPhase.HOST
                        pass
                    elif enum_equal(cmd, MGMT_COMMAND.GET_DATA) and \
                            enum_equal(phase, LoihiPhase.HOST):
                        # Handle get/set Var requests from runtime service
                        self._handle_get_var()
                    elif enum_equal(cmd,
                                    MGMT_COMMAND.SET_DATA) and \
                            enum_equal(phase, LoihiPhase.HOST):
                        # Handle get/set Var requests from runtime service
                        self._handle_set_var()
                    else:
                        raise ValueError(
                            f"Wrong Phase Info Received : {cmd}")
                except Exception as inst:
                    self.log.info(f"Exception {inst} occured while"
                                  f" running command {cmd} in {self.__class__}")
                    # Inform runtime service about termination
                    self.process_to_service.send(MGMT_RESPONSE.ERROR)
                    self.join()
                    raise inst
            else:
                # Handle VarPort requests from RefPorts
                self._handle_var_port(action)

            channel_actions = [(self.service_to_process, lambda: 'cmd')]
            if enum_equal(phase, LoihiPhase.PRE_MGMT) or \
                    enum_equal(phase, LoihiPhase.POST_MGMT):
                for var_port in self.var_ports:
                    for csp_port in var_port.csp_ports:
                        if isinstance(csp_port, CspRecvPort):
                            channel_actions.append(
                                (csp_port, lambda: var_port))
            action = selector.select(*channel_actions)

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
