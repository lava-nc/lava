# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy as np

from lava.magma.compiler.channels.pypychannel import CspSelector
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.mgmt_token_enums import (
    enum_to_np,
    enum_equal,
    MGMT_RESPONSE,
    MGMT_COMMAND,
)
from lava.magma.runtime.runtime_services.enums import (
    LoihiPhase,
    LoihiVersion
)
from lava.magma.runtime.runtime_services.interfaces import \
    AbstractRuntimeService

try:
    from nxsdk.arch.base.nxboard import NxBoard
except(ImportError):
    class NxBoard():
        pass

class PyRuntimeService(AbstractRuntimeService):
    pass


class CRuntimeService(AbstractRuntimeService):
    pass


class NcRuntimeService(AbstractRuntimeService):
    pass


class LoihiPyRuntimeService(PyRuntimeService):
    """RuntimeService that implements Loihi SyncProtocol in Python."""

    def _next_phase(self, curr_phase, is_last_time_step: bool):
        """Advances the current phase to the next phase.
        On the first time step it starts with HOST phase and advances to SPK.
        Afterwards it loops: SPK -> PRE_MGMT -> LRN -> POST_MGMT -> SPK
        On the last time step POST_MGMT advances to HOST phase."""
        if curr_phase == LoihiPhase.SPK:
            return LoihiPhase.PRE_MGMT
        elif curr_phase == LoihiPhase.PRE_MGMT:
            return LoihiPhase.LRN
        elif curr_phase == LoihiPhase.LRN:
            return LoihiPhase.POST_MGMT
        elif curr_phase == LoihiPhase.POST_MGMT and \
                is_last_time_step:
            return LoihiPhase.HOST
        elif curr_phase == LoihiPhase.POST_MGMT and not \
                is_last_time_step:
            return LoihiPhase.SPK
        elif curr_phase == LoihiPhase.HOST:
            return LoihiPhase.SPK

    def _send_pm_cmd(self, phase: MGMT_COMMAND):
        """Sends a command (phase information) to all ProcessModels."""
        for send_port in self.service_to_process:
            send_port.send(phase)

    def _send_pm_req_given_model_id(self, model_id: int, *requests):
        """Sends requests to a ProcessModel given by the model id."""
        process_idx = self.model_ids.index(model_id)
        req_port = self.service_to_process[process_idx]
        for request in requests:
            req_port.send(request)

    def _get_pm_resp(self) -> ty.Iterable[MGMT_RESPONSE]:
        """Retrieves responses of all ProcessModels."""
        rcv_msgs = []
        num_responses_expected = len(self.model_ids)
        counter = 0
        while counter < num_responses_expected:
            ptos_recv_port = self.process_to_service[counter]
            rcv_msgs.append(ptos_recv_port.recv())
            counter += 1
        return rcv_msgs

    def _relay_to_runtime_data_given_model_id(self, model_id: int):
        """Relays data received from ProcessModel given by model id  to the
        runtime"""
        process_idx = self.model_ids.index(model_id)
        data_recv_port = self.process_to_service[process_idx]
        data_relay_port = self.service_to_runtime
        num_items = data_recv_port.recv()
        data_relay_port.send(num_items)
        for i in range(int(num_items[0])):
            value = data_recv_port.recv()
            data_relay_port.send(value)

    def _relay_to_pm_data_given_model_id(self, model_id: int):
        """Relays data received from the runtime to the ProcessModel given by
        the model id."""
        process_idx = self.model_ids.index(model_id)

        data_recv_port = self.runtime_to_service
        data_relay_port = self.service_to_process[process_idx]
        # Receive and relay number of items
        num_items = data_recv_port.recv()
        data_relay_port.send(num_items)
        # Receive and relay data1, data2, ...
        for i in range(int(num_items[0].item())):
            data_relay_port.send(data_recv_port.recv())

    def _relay_pm_ack_given_model_id(self, model_id: int):
        """Relays ack received from ProcessModel given by model id to the
        runtime."""
        process_idx = self.model_ids.index(model_id)

        ack_recv_port = self.process_to_service[process_idx]
        ack_relay_port = self.service_to_runtime
        ack_relay_port.send(ack_recv_port.recv())

    def run(self):
        """Retrieves commands from the runtime. On STOP or PAUSE commands all
        ProcessModels are notified and expected to TERMINATE or PAUSE,
        respectively. Otherwise the number of time steps is received as command.
        In this case iterate through the phases of the Loihi protocol until the
        last time step is reached. The runtime is informed after the last time
        step. The loop ends when receiving the STOP command from the runtime."""
        selector = CspSelector()
        phase = LoihiPhase.HOST

        channel_actions = [(self.runtime_to_service, lambda: 'cmd')]

        while True:
            # Probe if there is a new command from the runtime
            action = selector.select(*channel_actions)

            if action == 'cmd':
                command = self.runtime_to_service.recv()
                if enum_equal(command, MGMT_COMMAND.STOP):
                    # Inform all ProcessModels about the STOP command
                    self._send_pm_cmd(command)
                    rsps = self._get_pm_resp()
                    for rsp in rsps:
                        if not enum_equal(rsp, MGMT_RESPONSE.TERMINATED):
                            raise ValueError(f"Wrong Response Received : {rsp}")
                    # Inform the runtime about successful termination
                    self.service_to_runtime.send(MGMT_RESPONSE.TERMINATED)
                    self.join()
                    return
                elif enum_equal(command, MGMT_COMMAND.PAUSE):
                    # Inform all ProcessModels about the PAUSE command
                    self._send_pm_cmd(command)
                    rsps = self._get_pm_resp()
                    for rsp in rsps:
                        if not enum_equal(rsp, MGMT_RESPONSE.PAUSED):
                            raise ValueError(f"Wrong Response Received : {rsp}")
                    # Inform the runtime about successful pausing
                    self.service_to_runtime.send(MGMT_RESPONSE.PAUSED)
                    break
                elif enum_equal(command, MGMT_COMMAND.GET_DATA) or \
                        enum_equal(command, MGMT_COMMAND.SET_DATA):
                    self._handle_get_set(phase, command)
                else:
                    # The number of time steps was received ("command")
                    # Start iterating through Loihi phases
                    curr_time_step = 0
                    phase = LoihiPhase.HOST
                    while True:
                        # Check if it is the last time step
                        is_last_ts = enum_equal(enum_to_np(curr_time_step),
                                                command)
                        # Advance to the next phase
                        phase = self._next_phase(phase, is_last_ts)
                        # Increase time step if spiking phase
                        if enum_equal(phase, LoihiPhase.SPK):
                            curr_time_step += 1
                        # Inform ProcessModels about current phase
                        self._send_pm_cmd(phase)
                        # ProcessModels respond with DONE if not HOST phase
                        if not enum_equal(
                                phase, LoihiPhase.HOST):

                            for rsp in self._get_pm_resp():
                                if not enum_equal(rsp, MGMT_RESPONSE.DONE):
                                    if enum_equal(rsp, MGMT_RESPONSE.ERROR):
                                        # Forward error to runtime
                                        self.service_to_runtime.send(
                                            MGMT_RESPONSE.ERROR)
                                        # stop all other pm
                                        self._send_pm_cmd(MGMT_COMMAND.STOP)
                                        return
                                    else:
                                        raise ValueError(
                                            f"Wrong Response Received : {rsp}")

                        # If HOST phase (last time step ended) break the loop
                        if enum_equal(
                                phase, LoihiPhase.HOST):
                            break

                    # Inform the runtime that last time step was reached
                    self.service_to_runtime.send(MGMT_RESPONSE.DONE)

    def _handle_get_set(self, phase, command):
        if enum_equal(phase, LoihiPhase.HOST):
            if enum_equal(command, MGMT_COMMAND.GET_DATA):
                requests: ty.List[np.ndarray] = [command]
                # recv model_id
                model_id: int = int(self.runtime_to_service.recv()[0].item())
                # recv var_id
                requests.append(self.runtime_to_service.recv())
                self._send_pm_req_given_model_id(model_id, *requests)
                self._relay_to_runtime_data_given_model_id(model_id)
            elif enum_equal(command, MGMT_COMMAND.SET_DATA):
                requests: ty.List[np.ndarray] = [command]
                # recv model_id
                model_id: int = int(self.runtime_to_service.recv()[0].item())
                # recv var_id
                requests.append(self.runtime_to_service.recv())
                self._send_pm_req_given_model_id(model_id, *requests)
                self._relay_to_pm_data_given_model_id(model_id)
            else:
                raise RuntimeError(f"Unknown request {command}")


class LoihiCRuntimeService(AbstractRuntimeService):
    """RuntimeService that implements Loihi SyncProtocol in C."""
    pass


class AsyncPyRuntimeService(PyRuntimeService):
    """RuntimeService that implements Async SyncProtocol in Py."""

    def _send_pm_cmd(self, cmd: MGMT_COMMAND):
        for stop_send_port in self.service_to_process:
            stop_send_port.send(cmd)

    def _get_pm_resp(self) -> ty.Iterable[MGMT_RESPONSE]:
        rcv_msgs = []
        for ptos_recv_port in self.process_to_service:
            rcv_msgs.append(ptos_recv_port.recv())
        return rcv_msgs

    def run(self):
        while True:
            command = self.runtime_to_service.recv()
            if enum_equal(command, MGMT_COMMAND.STOP):
                self._send_pm_cmd(command)
                rsps = self._get_pm_resp()
                for rsp in rsps:
                    if not enum_equal(rsp, MGMT_RESPONSE.TERMINATED):
                        raise ValueError(f"Wrong Response Received : {rsp}")
                self.service_to_runtime.send(MGMT_RESPONSE.TERMINATED)
                self.join()
                return
            else:
                self._send_pm_cmd(MGMT_COMMAND.RUN)
                rsps = self._get_pm_resp()
                for rsp in rsps:
                    if not enum_equal(rsp, MGMT_RESPONSE.DONE):
                        raise ValueError(f"Wrong Response Received : {rsp}")
                self.service_to_runtime.send(MGMT_RESPONSE.DONE)


class NxSDKRuntimeService(NcRuntimeService):
    """NxSDK RuntimeService that implements NxCore SyncProtocol.

    The NxSDKRuntimeService is a wrapper around NxCore that allows
    interaction with Loihi through NxCore API and GRPC communication
    channels to Loihi.

    Parameters
    ----------
    protocol: ty.Type[LoihiProtocol]
              Communication protocol used by NxSDKRuntimeService
    loihi_version: LoihiVersion
                   Version of Loihi Chip to use, N2 or N3
    """

    def __init__(self,
                 protocol: ty.Type[AbstractSyncProtocol],
                 loihi_version: LoihiVersion = LoihiVersion.N3,):
        super(NxSDKRuntimeService, self).__init__(
            protocol=protocol
        )
        self.board: NxBoard = None
        self.num_steps = 0

        if loihi_version == LoihiVersion.N3:
            from nxsdk.arch.n3b.n3board import N3Board
            # # TODO: Need to find good way to set Board Init
            self.board = N3Board(1, 1, [2], [[5, 5]])
        elif loihi_version == LoihiVersion.N2:
            from nxsdk.arch.n2a.n2board import N2Board # noqa F401
            self.board = N2Board(1, 1, [2], [[5, 5]])
        else:
            raise ValueError('Unsupported Loihi version '
                             + 'used in board selection')

    def _send_pm_cmd(self, cmd: MGMT_COMMAND):
        for stop_send_port in self.service_to_process:
            stop_send_port.send(cmd)

    def _send_pm_rn(self, run_number: int):
        for stop_send_port in self.service_to_process:
            stop_send_port.send(run_number)

    def _get_pm_resp(self) -> ty.Iterable[MGMT_RESPONSE]:
        rcv_msgs = []
        for ptos_recv_port in self.process_to_service:
            rcv_msgs.append(ptos_recv_port.recv())
        return rcv_msgs

    def run(self):
        self.num_steps = self.runtime_to_service.recv()
        self.service_to_runtime.send(MGMT_RESPONSE.DONE)

        selector = CspSelector()
        channel_actions = [(self.runtime_to_service, lambda: 'cmd')]

        while True:
            action = selector.select(*channel_actions)
            if action == 'cmd':
                command = self.runtime_to_service.recv()
                if enum_equal(command, MGMT_COMMAND.STOP):
                    self._send_pm_cmd(command)
                    rsps = self._get_pm_resp()
                    for rsp in rsps:
                        if not enum_equal(rsp, MGMT_RESPONSE.TERMINATED):
                            raise ValueError(f"Wrong Response Received : {rsp}")
                    self.service_to_runtime.send(MGMT_RESPONSE.TERMINATED)
                    self.join()
                    return
                elif enum_equal(command, MGMT_COMMAND.PAUSE):
                    self._send_pm_cmd(command)
                    rsps = self._get_pm_resp()
                    for rsp in rsps:
                        if not enum_equal(rsp, MGMT_RESPONSE.PAUSED):
                            raise ValueError(f"Wrong Response Received : {rsp}")

                    self.service_to_runtime.send(MGMT_RESPONSE.PAUSED)
                    break
                elif enum_equal(command, MGMT_COMMAND.RUN):
                    self._send_pm_cmd(MGMT_COMMAND.RUN)
                    rsps = self._get_pm_resp()
                    self._send_pm_cmd(self.num_steps)
                    rsps = rsps + self._get_pm_resp()
                    for rsp in rsps:
                        if not enum_equal(rsp, MGMT_RESPONSE.DONE):
                            raise ValueError(f"Wrong Response Received : {rsp}")
                    self.service_to_runtime.send(MGMT_RESPONSE.DONE)
                else:
                    self.service_to_runtime.send(MGMT_RESPONSE.ERROR)

                    self._send_pm_cmd(MGMT_COMMAND.STOP)
                    return

    def get_board(self) -> NxBoard:
        if self.board is not None:
            return self.board
        else:
            AssertionError("Cannot return board, self.board is None")
