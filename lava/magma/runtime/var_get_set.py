# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
import numpy as np

from lava.magma.compiler.channels.interfaces import AbstractCspRecvPort, \
    AbstractCspSendPort
from lava.magma.runtime.mgmt_token_enums import SynchronizationPhase, REQ_TYPE


class VarGetSet:
    """Handler class to relay get or set messages for vars"""
    def __init__(self,
                 runtime_to_service_req: AbstractCspRecvPort,
                 model_ids: ty.List[int],
                 service_to_runtime_ack: AbstractCspSendPort,
                 service_to_runtime_data: AbstractCspSendPort,
                 runtime_to_service_data: AbstractCspRecvPort,
                 process_to_service_ack: ty.Iterable[AbstractCspRecvPort],
                 service_to_process_req: ty.Iterable[AbstractCspSendPort],
                 process_to_service_data: ty.Iterable[AbstractCspRecvPort],
                 service_to_process_data: ty.Iterable[AbstractCspSendPort],
                 ):
        self.runtime_to_service_req = runtime_to_service_req
        self.model_ids: ty.List[int] = model_ids
        self.service_to_runtime_ack: AbstractCspSendPort = \
            service_to_runtime_ack
        self.service_to_runtime_data: AbstractCspSendPort = \
            service_to_runtime_data
        self.runtime_to_service_data: AbstractCspRecvPort = \
            runtime_to_service_data
        self.process_to_service_ack: ty.Iterable[AbstractCspRecvPort] = \
            process_to_service_ack
        self.service_to_process_req: ty.Iterable[AbstractCspSendPort] = \
            service_to_process_req
        self.process_to_service_data: ty.Iterable[AbstractCspRecvPort] = \
            process_to_service_data
        self.service_to_process_data: ty.Iterable[AbstractCspSendPort] = \
            service_to_process_data

    def get(self, request: np.ndarray):
        """Get Var given the request"""
        requests: ty.List[np.ndarray] = [request]
        # recv model_id
        model_id: int = self.runtime_to_service_req.recv()[0].item()
        # recv var_id
        requests.append(self.runtime_to_service_req.recv())
        self._send_pm_req_given_model_id(model_id,*requests)

        self._relay_to_runtime_data_given_model_id(
            model_id)

    def set(self, request: np.ndarray):
        """Set Var given the request"""
        requests: ty.List[np.ndarray] = [request]
        # recv model_id
        model_id: int = self.runtime_to_service_req.recv()[0].item()
        # recv var_id
        requests.append(self.runtime_to_service_req.recv())
        self._send_pm_req_given_model_id(model_id, *requests)

        self._relay_to_pm_data_given_model_id(model_id)

    def _relay_pm_ack_given_model_id(self, model_id: int):
        """Relays ack received from pm to runtime"""
        process_idx: int = self.model_ids.index(model_id)

        ack_recv_port: AbstractCspRecvPort = \
            self.process_to_service_ack[process_idx]
        ack_relay_port: AbstractCspSendPort = self.service_to_runtime_ack
        ack_relay_port.send(ack_recv_port.recv())

    def _relay_to_pm_data_given_model_id(self, model_id: int):
        """Relays data received from runtime to pm"""
        process_idx: int = self.model_ids.index(model_id)

        data_recv_port: AbstractCspRecvPort = self.runtime_to_service_data
        data_relay_port: AbstractCspSendPort = \
            self.service_to_process_data[process_idx]
        # recv and relay num_items
        num_items: np.ndarray = data_recv_port.recv()
        data_relay_port.send(num_items)
        # recv and relay data1, data2, ...
        for i in range(num_items[0].item()):
            data_relay_port.send(data_recv_port.recv())

    def _relay_to_runtime_data_given_model_id(self, model_id: int):
        """Relays data received from pm to runtime"""
        process_idx: int = self.model_ids.index(model_id)

        data_recv_port: AbstractCspRecvPort = \
            self.process_to_service_data[process_idx]
        data_relay_port: AbstractCspSendPort = self.service_to_runtime_data
        num_items: np.ndarray = data_recv_port.recv()
        data_relay_port.send(num_items)
        for i in range(num_items[0]):
            data_relay_port.send(data_recv_port.recv())

    def _send_pm_req_given_model_id(self, model_id: int, *requests):
        process_idx: int = self.model_ids.index(model_id)
        req_port: AbstractCspSendPort = self.service_to_process_req[process_idx]
        for request in requests:
            req_port.send(request)

    def handle_var_get_set(self, phase: np.ndarray):
        """Handle var get set"""
        if np.array_equal(phase, SynchronizationPhase.PRE_MGMT) or \
                np.array_equal(phase, SynchronizationPhase.POST_MGMT):
            while self.runtime_to_service_req.probe():
                request = self.runtime_to_service_req.recv()
                if np.array_equal(request, REQ_TYPE.GET):
                    self.get(request)
                elif np.array_equal(request, REQ_TYPE.SET):
                    self.set(request)
                else:
                    raise RuntimeError(f"Unknown request {request}")
