# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC, abstractmethod

import numpy as np

from lava.magma.compiler.channels.pypychannel import CspSendPort, CspRecvPort
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.py.ports import AbstractPyPort
from lava.magma.runtime.mgmt_token_enums import (
    enum_to_message,
    enum_to_np,
    MGMT_COMMAND,
    MGMT_RESPONSE, REQ_TYPE,
)


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
        self.service_to_process_cmd: ty.Optional[CspRecvPort] = None
        self.process_to_service_ack: ty.Optional[CspSendPort] = None
        self.service_to_process_req: ty.Optional[CspRecvPort] = None
        self.process_to_service_data: ty.Optional[CspSendPort] = None
        self.service_to_process_data: ty.Optional[CspRecvPort] = None
        self.py_ports: ty.List[AbstractPyPort] = []
        self.var_id_to_var_map: ty.Dict[int, ty.Any] = {}

    def __setattr__(self, key: str, value: ty.Any):
        self.__dict__[key] = value
        if isinstance(value, AbstractPyPort):
            self.py_ports.append(value)

    def start(self):
        self.service_to_process_cmd.start()
        self.process_to_service_ack.start()
        self.service_to_process_req.start()
        self.process_to_service_data.start()
        self.service_to_process_data.start()
        for p in self.py_ports:
            p.start()
        self.run()

    @abstractmethod
    def run(self):
        pass

    def join(self):
        self.service_to_process_cmd.join()
        self.process_to_service_ack.join()
        self.service_to_process_req.join()
        self.process_to_service_data.join()
        self.service_to_process_data.join()
        for p in self.py_ports:
            p.join()


class PyLoihiProcessModel(AbstractPyProcessModel):
    def __init__(self):
        super(PyLoihiProcessModel, self).__init__()
        self.current_ts = 0

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

    def run_host_mgmt(self):
        pass

    def pre_guard(self):
        pass

    def lrn_guard(self):
        pass

    def post_guard(self):
        pass

    def host_guard(self):
        pass

    def run(self):
        while True:
            if self.service_to_process_cmd.probe():
                phase = self.service_to_process_cmd.recv()
                if np.array_equal(phase, MGMT_COMMAND.STOP):
                    self.process_to_service_ack.send(
                        enum_to_message(MGMT_RESPONSE.TERMINATED)
                    )
                    self.join()
                    return
                if np.array_equal(phase, PyLoihiProcessModel.Phase.SPK):
                    self.current_ts += 1
                    self.run_spk()
                elif np.array_equal(phase, PyLoihiProcessModel.Phase.PRE_MGMT):
                    if self.pre_guard():
                        self.run_pre_mgmt()
                    self._handle_get_set_var()
                elif np.array_equal(phase, PyLoihiProcessModel.Phase.LRN):
                    if self.lrn_guard():
                        self.run_lrn()
                elif np.array_equal(phase, PyLoihiProcessModel.Phase.POST_MGMT):
                    if self.post_guard():
                        self.run_post_mgmt()
                    self._handle_get_set_var()
                elif np.array_equal(phase, PyLoihiProcessModel.Phase.HOST):
                    if self.host_guard():
                        self.run_host_mgmt()
                else:
                    raise ValueError(f"Wrong Phase Info Received : {phase}")
                self.process_to_service_ack.send(
                    enum_to_message(MGMT_RESPONSE.DONE)
                )
            else:
                self._handle_get_set_var()

    def _handle_get_set_var(self):
        while self.service_to_process_req.probe():
            req_port: CspRecvPort = self.service_to_process_req
            request: np.ndarray = req_port.recv()
            if np.array_equal(request, REQ_TYPE.GET):
                self._handle_get_var()
            elif np.array_equal(request, REQ_TYPE.SET):
                self._handle_set_var()
            else:
                raise RuntimeError(f"Unknown request type {request}")

    def _handle_get_var(self):
        # 1. Recv Var ID
        req_port: CspRecvPort = self.service_to_process_req
        var_id: int = req_port.recv()[0].item()
        var_name: str = self.var_id_to_var_map[var_id]
        var: ty.Any = getattr(self, var_name)

        # 2. Send Var data
        data_port: CspSendPort = self.process_to_service_data
        if isinstance(var, int) or isinstance(var, np.integer):
            data_port.send(enum_to_message(1))
            data_port.send(enum_to_message(var))
        elif isinstance(var, np.ndarray):
            var_iter = np.nditer(var)
            num_items: np.integer = np.prod(var.shape)
            data_port.send(enum_to_message(num_items))
            for value in var_iter:
                data_port.send(enum_to_message(value))

    def _handle_set_var(self):
        # 1. Recv Var ID
        req_port: CspRecvPort = self.service_to_process_req
        var_id: int = req_port.recv()[0].item()
        var_name: str = self.var_id_to_var_map[var_id]
        var: ty.Any = getattr(self, var_name)

        # 2. Recv Var data
        data_port: CspRecvPort = self.service_to_process_data
        if isinstance(var, int) or isinstance(var, np.integer):
            data_port.recv()  # Ignore as this will be 1 (num_items)
            buffer = data_port.recv()[0]
            if isinstance(var, int):
                setattr(self, var_name, buffer.item())
            else:
                setattr(self, var_name, buffer.astype(var.dtype))
        elif isinstance(var, np.ndarray):
            num_items = data_port.recv()[0]
            var_iter = np.nditer(var, op_flags=['readwrite'])

            for i in var_iter:
                if num_items == 0:
                    break
                num_items -= 1
                i[...] = data_port.recv()[0]
        else:
            raise RuntimeError("Unsupported type")
