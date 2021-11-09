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
    enum_to_np,
    SynchronizationPhase,
)
from lava.magma.runtime.synchronizer import AbstractSynchronizer


class RuntimeService(ABC):
    def __init__(self,
                 protocol: AbstractSyncProtocol,
                 compute_resource_type: ty.Type):
        self.protocol: AbstractSyncProtocol = protocol
        self.compute_resource_type: ty.Type = compute_resource_type
        self.synchronizer: AbstractSynchronizer = \
            self.protocol.synchronizer[compute_resource_type](protocol)

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
        return f"Synchronizer : {self.synchronizer}, \
                 RuntimeServiceId : {self.runtime_service_id}, \
                 Protocol: {self.protocol}, \
                 ComputeResourceType: {self.compute_resource_type}"

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

    def run(self):
        self.synchronizer.reset_phase()
        while True:
            if self.runtime_to_service_cmd.probe():
                command = self.runtime_to_service_cmd.recv()
                if np.array_equal(command, MGMT_COMMAND.STOP):
                    self.synchronizer.stop()
                    self.service_to_runtime_ack.send(MGMT_RESPONSE.TERMINATED)
                    self.join()
                    return
                elif np.array_equal(command, MGMT_COMMAND.PAUSE):
                    self.synchronizer.pause()
                    self.service_to_runtime_ack.send(MGMT_RESPONSE.PAUSED)
                    break
                else:
                    curr_time_step = 0
                    self.synchronizer.reset_phase()
                    while not np.array_equal(enum_to_np(curr_time_step),
                                             command):
                        phase = self.synchronizer.phase
                        if np.array_equal(phase, SynchronizationPhase.SPK):
                            curr_time_step += 1
                        is_last_ts = np.array_equal(enum_to_np(curr_time_step),
                                                    command)
                        is_last_phase = \
                            np.array_equal(phase,
                                           SynchronizationPhase.POST_MGMT)
                        auto_increment_phase = True
                        if is_last_ts and is_last_phase:
                            auto_increment_phase = False
                        self.synchronizer.run(auto_increment_phase)

                    self.service_to_runtime_ack.send(MGMT_RESPONSE.DONE)

            self._handle_get_set(self.synchronizer.phase)


# class AsyncPyRuntimeService(PyRuntimeService):
#     """RuntimeService that implements Async SyncProtocol in Py."""
#
#     def _send_pm_cmd(self, cmd: MGMT_COMMAND):
#         for stop_send_port in self.service_to_process_cmd:
#             stop_send_port.send(cmd)
#
#     def _get_pm_resp(self) -> ty.Iterable[MGMT_RESPONSE]:
#         rcv_msgs = []
#         for ptos_recv_port in self.process_to_service_ack:
#             rcv_msgs.append(ptos_recv_port.recv())
#
#         return rcv_msgs
#
#     def run(self):
#         while True:
#             command = self.runtime_to_service_cmd.recv()
#             if np.array_equal(command, MGMT_COMMAND.STOP):
#                 self._send_pm_cmd(command)
#                 rsps = self._get_pm_resp()
#                 for rsp in rsps:
#                     if not np.array_equal(rsp, MGMT_RESPONSE.TERMINATED):
#                         raise ValueError(f"Wrong Response Received : {rsp}")
#                 self.service_to_runtime_ack.send(MGMT_RESPONSE.TERMINATED)
#                 self.join()
#                 return
#             else:
#                 self._send_pm_cmd(MGMT_COMMAND.RUN)
#                 rsps = self._get_pm_resp()
#                 for rsp in rsps:
#                     if not np.array_equal(rsp, MGMT_RESPONSE.DONE):
#                         raise ValueError(f"Wrong Response Received : {rsp}")
#                 self.service_to_runtime_ack.send(MGMT_RESPONSE.DONE)
