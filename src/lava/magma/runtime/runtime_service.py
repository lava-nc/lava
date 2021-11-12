# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from abc import ABC, abstractmethod

import numpy as np

from lava.magma.compiler.channels.pypychannel import CspRecvPort, CspSendPort
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.mgmt_token_enums import (
    MGMT_RESPONSE,
    MGMT_COMMAND,
    enum_to_np, REQ_TYPE,
)


class AbstractRuntimeService(ABC):
    def __init__(self, protocol):
        self.protocol: ty.Optional[AbstractSyncProtocol] = protocol

        self.runtime_service_id: ty.Optional[int] = None

        self.runtime_to_service_cmd: ty.Optional[CspRecvPort] = None
        self.service_to_runtime_ack: ty.Optional[CspSendPort] = None
        self.runtime_to_service_req: ty.Optional[CspRecvPort] = None
        self.service_to_runtime_data: ty.Optional[CspSendPort] = None
        self.runtime_to_service_data: ty.Optional[CspRecvPort] = None

        self.model_ids: ty.List[int] = []

        self.service_to_process_cmd: ty.Iterable[CspSendPort] = []
        self.process_to_service_ack: ty.Iterable[CspRecvPort] = []
        self.service_to_process_req: ty.Iterable[CspSendPort] = []
        self.process_to_service_data: ty.Iterable[CspRecvPort] = []
        self.service_to_process_data: ty.Iterable[CspSendPort] = []

    def __repr__(self):
        return f"Synchronizer : {self.__class__}, \
                 RuntimeServiceId : {self.runtime_service_id}, \
                 Protocol: {self.protocol}"

    def start(self):
        self.runtime_to_service_cmd.start()
        self.service_to_runtime_ack.start()
        self.runtime_to_service_req.start()
        self.service_to_runtime_data.start()
        self.runtime_to_service_data.start()
        for i in range(len(self.service_to_process_cmd)):
            self.service_to_process_cmd[i].start()
            self.process_to_service_ack[i].start()
            self.service_to_process_req[i].start()
            self.process_to_service_data[i].start()
            self.service_to_process_data[i].start()
        self.run()

    @abstractmethod
    def run(self):
        pass

    def join(self):
        self.runtime_to_service_cmd.join()
        self.service_to_runtime_ack.join()
        self.runtime_to_service_req.join()
        self.service_to_runtime_data.join()
        self.runtime_to_service_data.join()

        for i in range(len(self.service_to_process_cmd)):
            self.service_to_process_cmd[i].join()
            self.process_to_service_ack[i].join()
            self.service_to_process_req[i].join()
            self.process_to_service_data[i].join()
            self.service_to_process_data[i].join()


class PyRuntimeService(AbstractRuntimeService):
    pass


class CRuntimeService(AbstractRuntimeService):
    pass


class LoihiPyRuntimeService(PyRuntimeService):
    """RuntimeService that implements Loihi SyncProtocol in Python."""

    class Phase:
        SPK = enum_to_np(1)
        PRE_MGMT = enum_to_np(2)
        LRN = enum_to_np(3)
        POST_MGMT = enum_to_np(4)
        HOST = enum_to_np(5)

    def _next_phase(self, curr_phase, is_last_time_step: bool):
        if curr_phase == LoihiPyRuntimeService.Phase.SPK:
            return LoihiPyRuntimeService.Phase.PRE_MGMT
        elif curr_phase == LoihiPyRuntimeService.Phase.PRE_MGMT:
            return LoihiPyRuntimeService.Phase.LRN
        elif curr_phase == LoihiPyRuntimeService.Phase.LRN:
            return LoihiPyRuntimeService.Phase.POST_MGMT
        elif curr_phase == LoihiPyRuntimeService.Phase.POST_MGMT and \
                is_last_time_step:
            return LoihiPyRuntimeService.Phase.HOST
        elif curr_phase == LoihiPyRuntimeService.Phase.POST_MGMT and not \
                is_last_time_step:
            return LoihiPyRuntimeService.Phase.SPK
        elif curr_phase == LoihiPyRuntimeService.Phase.HOST:
            return LoihiPyRuntimeService.Phase.SPK

    def _send_pm_cmd(self, phase: MGMT_COMMAND):
        for send_port in self.service_to_process_cmd:
            send_port.send(phase)

    def _send_pm_req_given_model_id(self, model_id: int, *requests):
        process_idx: int = self.model_ids.index(model_id)
        req_port: CspSendPort = self.service_to_process_req[process_idx]
        for request in requests:
            req_port.send(request)

    def _get_pm_resp(self) -> ty.Iterable[MGMT_RESPONSE]:
        rcv_msgs = []
        num_responses_expected: int = len(self.model_ids)
        counter: int = 0
        while counter < num_responses_expected:
            ptos_recv_port = self.process_to_service_ack[counter]
            if ptos_recv_port.probe():
                rcv_msgs.append(ptos_recv_port.recv())
                counter += 1
        return rcv_msgs

    def _relay_to_runtime_data_given_model_id(self, model_id: int):
        """Relays data received from pm to runtime"""
        process_idx: int = self.model_ids.index(model_id)

        data_recv_port: CspRecvPort = self.process_to_service_data[process_idx]
        data_relay_port: CspSendPort = self.service_to_runtime_data
        num_items: np.ndarray = data_recv_port.recv()
        data_relay_port.send(num_items)
        for i in range(num_items[0]):
            data_relay_port.send(data_recv_port.recv())

    def _relay_to_pm_data_given_model_id(self, model_id: int):
        """Relays data received from runtime to pm"""
        process_idx: int = self.model_ids.index(model_id)

        data_recv_port: CspRecvPort = self.runtime_to_service_data
        data_relay_port: CspSendPort = self.service_to_process_data[process_idx]
        # recv and relay num_items
        num_items: np.ndarray = data_recv_port.recv()
        data_relay_port.send(num_items)
        # recv and relay data1, data2, ...
        for i in range(num_items[0].item()):
            data_relay_port.send(data_recv_port.recv())

    def _relay_pm_ack_given_model_id(self, model_id: int):
        """Relays ack received from pm to runtime"""
        process_idx: int = self.model_ids.index(model_id)

        ack_recv_port: CspRecvPort = self.process_to_service_ack[process_idx]
        ack_relay_port: CspSendPort = self.service_to_runtime_ack
        ack_relay_port.send(ack_recv_port.recv())

    def run(self):
        phase = LoihiPyRuntimeService.Phase.HOST
        while True:
            if self.runtime_to_service_cmd.probe():
                command = self.runtime_to_service_cmd.recv()
                if np.array_equal(command, MGMT_COMMAND.STOP):
                    self._send_pm_cmd(command)
                    rsps = self._get_pm_resp()
                    for rsp in rsps:
                        if not np.array_equal(rsp, MGMT_RESPONSE.TERMINATED):
                            raise ValueError(f"Wrong Response Received : {rsp}")
                    self.service_to_runtime_ack.send(MGMT_RESPONSE.TERMINATED)
                    self.join()
                    return
                elif np.array_equal(command, MGMT_COMMAND.PAUSE):
                    self._send_pm_cmd(command)
                    rsps = self._get_pm_resp()
                    for rsp in rsps:
                        if not np.array_equal(rsp, MGMT_RESPONSE.PAUSED):
                            raise ValueError(f"Wrong Response Received : {rsp}")
                    self.service_to_runtime_ack.send(MGMT_RESPONSE.PAUSED)
                    break
                else:
                    curr_time_step = 0
                    phase = LoihiPyRuntimeService.Phase.HOST
                    while True:
                        is_last_ts = np.array_equal(enum_to_np(curr_time_step),
                                                    command)
                        phase = self._next_phase(phase, is_last_ts)
                        if np.array_equal(phase,
                                          LoihiPyRuntimeService.Phase.SPK):
                            curr_time_step += 1
                        self._send_pm_cmd(phase)
                        if not np.array_equal(
                                phase, LoihiPyRuntimeService.Phase.HOST):
                            rsps = self._get_pm_resp()
                            for rsp in rsps:
                                if not np.array_equal(rsp, MGMT_RESPONSE.DONE):
                                    raise ValueError(
                                        f"Wrong Response Received : {rsp}")

                        if np.array_equal(
                                phase, LoihiPyRuntimeService.Phase.HOST):
                            break

                    self.service_to_runtime_ack.send(MGMT_RESPONSE.DONE)

            self._handle_get_set(phase)

    def _handle_get_set(self, phase):
        if np.array_equal(phase, LoihiPyRuntimeService.Phase.HOST):
            while True:
                if self.runtime_to_service_req.probe():
                    request = self.runtime_to_service_req.recv()
                    if np.array_equal(request, REQ_TYPE.GET):
                        requests: ty.List[np.ndarray] = [request]
                        # recv model_id
                        model_id: int = \
                            self.runtime_to_service_req.recv()[
                                0].item()
                        # recv var_id
                        requests.append(
                            self.runtime_to_service_req.recv())
                        self._send_pm_req_given_model_id(model_id,
                                                         *requests)

                        self._relay_to_runtime_data_given_model_id(
                            model_id)
                    elif np.array_equal(request, REQ_TYPE.SET):
                        requests: ty.List[np.ndarray] = [request]
                        # recv model_id
                        model_id: int = \
                            self.runtime_to_service_req.recv()[
                                0].item()
                        # recv var_id
                        requests.append(
                            self.runtime_to_service_req.recv())
                        self._send_pm_req_given_model_id(model_id,
                                                         *requests)

                        self._relay_to_pm_data_given_model_id(
                            model_id)
                    else:
                        raise RuntimeError(
                            f"Unknown request {request}")

                if self.runtime_to_service_cmd.probe():
                    return


class LoihiCRuntimeService(AbstractRuntimeService):
    """RuntimeService that implements Loihi SyncProtocol in C."""
    pass


class AsyncPyRuntimeService(PyRuntimeService):
    """RuntimeService that implements Async SyncProtocol in Py."""

    def _send_pm_cmd(self, cmd: MGMT_COMMAND):
        for stop_send_port in self.service_to_process_cmd:
            stop_send_port.send(cmd)

    def _get_pm_resp(self) -> ty.Iterable[MGMT_RESPONSE]:
        rcv_msgs = []
        for ptos_recv_port in self.process_to_service_ack:
            rcv_msgs.append(ptos_recv_port.recv())

        return rcv_msgs

    def run(self):
        while True:
            command = self.runtime_to_service_cmd.recv()
            if np.array_equal(command, MGMT_COMMAND.STOP):
                self._send_pm_cmd(command)
                rsps = self._get_pm_resp()
                for rsp in rsps:
                    if not np.array_equal(rsp, MGMT_RESPONSE.TERMINATED):
                        raise ValueError(f"Wrong Response Received : {rsp}")
                self.service_to_runtime_ack.send(MGMT_RESPONSE.TERMINATED)
                self.join()
                return
            else:
                self._send_pm_cmd(MGMT_COMMAND.RUN)
                rsps = self._get_pm_resp()
                for rsp in rsps:
                    if not np.array_equal(rsp, MGMT_RESPONSE.DONE):
                        raise ValueError(f"Wrong Response Received : {rsp}")
                self.service_to_runtime_ack.send(MGMT_RESPONSE.DONE)
