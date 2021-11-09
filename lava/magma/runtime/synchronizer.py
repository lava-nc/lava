# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from abc import ABC, abstractmethod
from enum import IntEnum
import numpy as np
import typing as ty

from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.mgmt_token_enums import SynchronizationPhase, \
    MGMT_COMMAND, MGMT_RESPONSE
from lava.magma.runtime.var_get_set import VarGetSet


class SynchronizerType(IntEnum):
    SYNC = 0
    ASYNC = 1


class AbstractSynchronizer(ABC):
    def __init__(self, protocol=AbstractSyncProtocol):
        self.protocol = protocol

    @property
    @abstractmethod
    def phase(self) -> SynchronizationPhase:
        pass

    def reset_phase(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def pause(self):
        pass

    @abstractmethod
    def run(self, auto_increment_phase=True):
        pass

    def var_get_set_handler(self):
        pass


class LoihiSynchronizer(AbstractSynchronizer):
    def __init__(self, protocol=AbstractSyncProtocol):
        super().__init__(protocol=protocol)
        self._phase = SynchronizationPhase.SPK
        self._var_get_set_handler: ty.Optional[VarGetSet] = None


    def var_get_set_handler(self):
        return self._var_get_set_handler

    @property
    def phase(self) -> np.ndarray:
        return self._phase

    @staticmethod
    def _next_phase(curr_phase: SynchronizationPhase):
        if curr_phase == SynchronizationPhase.SPK:
            return SynchronizationPhase.PRE_MGMT
        elif curr_phase == SynchronizationPhase.PRE_MGMT:
            return SynchronizationPhase.LRN
        elif curr_phase == SynchronizationPhase.LRN:
            return SynchronizationPhase.POST_MGMT
        elif curr_phase == SynchronizationPhase.POST_MGMT:
            return SynchronizationPhase.HOST
        elif curr_phase == SynchronizationPhase.HOST:
            return SynchronizationPhase.SPK

    def _send_pm_cmd(self, command: np.ndarray):
        for send_port in self.service_to_process_cmd:
            send_port.send(command)

    def _get_pm_resp(self, phase: np.ndarray) \
            -> ty.Iterable[np.ndarray]:
        rcv_msgs = []
        num_responses_expected: int = len(self.model_ids)
        counter: int = 0
        while counter < num_responses_expected:
            ptos_recv_port = self.process_to_service_ack[counter]
            self._handle_get_set(phase)
            if ptos_recv_port.probe():
                rcv_msgs.append(ptos_recv_port.recv())
                counter += 1
        return rcv_msgs

    def stop(self):
        self._send_pm_cmd(MGMT_COMMAND.STOP)
        rsps = self._get_pm_resp(self.phase)
        for rsp in rsps:
            if not np.array_equal(rsp, MGMT_RESPONSE.TERMINATED):
                raise ValueError(f"Wrong Response Received : {rsp}")

    def pause(self):
        self._send_pm_cmd(MGMT_COMMAND.PAUSE)
        rsps: ty.Iterable[np.ndarray] = self._get_pm_resp(self.phase)
        for rsp in rsps:
            if not np.array_equal(rsp, MGMT_RESPONSE.PAUSED):
                raise ValueError(f"Wrong Response Received : {rsp}")

    def reset_phase(self):
        self._phase = SynchronizationPhase.SPK

    def run(self, auto_increment_phase=True):
        self._send_pm_cmd(self._phase)
        rsps = self._get_pm_resp(self._phase)
        for rsp in rsps:
            if not np.array_equal(rsp, MGMT_RESPONSE.DONE):
                raise ValueError(
                    f"Wrong Response Received : {rsp}")
        if auto_increment_phase:
            self._phase = self._next_phase(self._phase)

