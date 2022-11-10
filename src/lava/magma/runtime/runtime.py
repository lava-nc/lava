# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: LGPL 2.1 or later
# See: https://spdx.org/licenses/
from __future__ import annotations

import logging
import sys
import traceback
import typing
import typing as ty
import numpy as np
from message_infrastructure import (RecvPort,
                                    SendPort,
                                    Actor,
                                    Channel)

from lava.magma.compiler.var_model import AbstractVarModel, LoihiSynapseVarModel
from message_infrastructure.message_interface_enum import ActorType
from message_infrastructure.factory import MessageInfrastructureFactory
from message_infrastructure.message_infrastructure_interface import \
    MessageInfrastructureInterface
from lava.magma.runtime.mgmt_token_enums import (MGMT_COMMAND, MGMT_RESPONSE,
                                                 enum_equal, enum_to_np)
from lava.magma.runtime.runtime_services.runtime_service import \
    AsyncPyRuntimeService

if ty.TYPE_CHECKING:
    from lava.magma.core.process.process import AbstractProcess
from lava.magma.compiler.builders.channel_builder import (
    ChannelBuilderMp, RuntimeChannelBuilderMp, ServiceChannelBuilderMp)
from lava.magma.compiler.builders.interfaces import AbstractProcessBuilder
from lava.magma.compiler.builders.py_builder import PyProcessBuilder
from lava.magma.compiler.builders.runtimeservice_builder import \
    RuntimeServiceBuilder
from lava.magma.compiler.executable import Executable
from lava.magma.compiler.node import NodeConfig
from lava.magma.core.process.ports.ports import create_port_id
from lava.magma.core.run_conditions import (AbstractRunCondition,
                                            RunContinuous, RunSteps)

"""Defines a Runtime which takes a lava executable and a pluggable message
passing infrastructure (for instance multiprocessing+shared memory or ray in
future), builds the components of the executable populated by the compiler
and starts the execution. Runtime is also responsible for auxiliary actions
such as pause, stop, wait (non-blocking run) etc.

Overall Runtime Architecture:
                                                                    (c) InVar/
                                                                        OutVar/
                                                                        RefVar
                                                                         _____
        (c) runtime_to_service                (c) service_to_process     |   |
        --------------------->                --------------------->     |   V
(s) Runtime                 (*s) RuntimeService             (*s) Process Models
        <---------------------                <---------------------
        (c) service_to_runtime                (c) process_to_service

(s) - Service
(c) - Channel
(*) - Multiple

Runtime coordinates with multiple RuntimeServices depending on how many got
created. The number of RuntimeServices is determined at compile time based
on the RunConfiguration supplied to the compiler.

Each RuntimeService is assigned a group of process models it is supposed to
manage. Actions/Commands issued by the Runtime are relayed to the
RuntimeService using the runtime_to_service channel and the response are
returned back using the service_to_runtime channel.

The RuntimeService further takes this forward for each process model in
similar fashion. A RuntimeService is connected to the process model it is
coordinating by two channels - service_to_process for sending
actions/commands to process model and process_to_service to get response back
from process model.

Process Models communicate with each other via channels defined by
InVar/OutVar/RefVar ports.
"""


def target_fn(*args, **kwargs):
    """
    Function to build and attach a system process to

    :param args: List Parameters to be passed onto the process
    :param kwargs: Dict Parameters to be passed onto the process
    :return: None
    """
    try:
        builder = kwargs.pop("builder")
        actor = builder.build()
        actor.start(*args, **kwargs)
    except Exception as e:
        print("Encountered Fatal Exception: " + str(e))
        print("Traceback: ")
        print(traceback.format_exc())
        raise e


class Runtime:
    """Lava runtime which consumes an executable and run
    run_condition. Exposes
    the APIs to start, pause, stop and wait on an execution. Execution could
    be blocking and non-blocking as specified by the run
    run_condition."""

    def __init__(self,
                 exe: Executable,
                 message_infrastructure_type: ActorType,
                 loglevel: int = logging.WARNING):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(loglevel)
        self._run_cond: typing.Optional[AbstractRunCondition] = None
        self._executable: Executable = exe

        self._messaging_infrastructure_type: ActorType = \
            message_infrastructure_type
        self._messaging_infrastructure: \
            ty.Optional[MessageInfrastructureInterface] = None
        self._is_initialized: bool = False
        self._is_running: bool = False
        self._is_started: bool = False
        self._req_paused: bool = False
        self._req_stop: bool = False
        self.runtime_to_service: ty.Iterable[SendPort] = []
        self.service_to_runtime: ty.Iterable[RecvPort] = []

    def __del__(self):
        """On destruction, terminate Runtime automatically to
        free compute resources.
        """
        if self._is_started:
            self.stop()

    def initialize(self, node_cfg_idx: int = 0):
        """Initializes the runtime"""
        self._build_message_infrastructure()
        self._build_channels()
        self._build_sync_channels()
        self._build_processes()
        self._build_runtime_services()
        self._start_ports()
        self.log.debug("Runtime Initialization Complete")
        self._is_initialized = True

    def _start_ports(self):
        """Start the ports of the runtime to communicate with runtime
        services"""
        for port in self.runtime_to_service:
            port.start()
        for port in self.service_to_runtime:
            port.start()

    @property
    def node_cfg(self) -> ty.List[NodeConfig]:
        """Returns the selected NodeCfg."""
        return self._executable.node_configs

    def _build_message_infrastructure(self):
        """Create the Messaging Infrastructure Backend given the
        _messaging_infrastructure_type and Start it"""
        self._messaging_infrastructure = MessageInfrastructureFactory.create(
            self._messaging_infrastructure_type)

    def _get_process_builder_for_process(self, process: AbstractProcess) -> \
            AbstractProcessBuilder:
        """Given a process return its process builder."""
        return self._executable.proc_builders[process]

    def _build_channels(self):
        """Given the channel builders for an executable,
        build these channels"""
        if self._executable.channel_builders:
            for channel_builder in self._executable.channel_builders:
                if isinstance(channel_builder, ChannelBuilderMp):
                    channel = channel_builder.build(
                        self._messaging_infrastructure
                    )

                    self._get_process_builder_for_process(
                        channel_builder.src_process).set_csp_ports(
                        [channel.src_port])

                    dst_pb = self._get_process_builder_for_process(
                        channel_builder.dst_process)
                    dst_pb.set_csp_ports([channel.dst_port])

                    # Add a mapping from the ID of the source PyPort
                    # to the CSP port
                    src_port_id = create_port_id(
                        channel_builder.src_process.id,
                        channel_builder.src_port_initializer.name)
                    dst_pb.add_csp_port_mapping(src_port_id, channel.dst_port)

    def _build_sync_channels(self):
        """Builds the channels needed for synchronization between runtime
        components"""
        if self._executable.sync_channel_builders:
            for sync_channel_builder in self._executable.sync_channel_builders:
                channel: Channel = sync_channel_builder.build(
                    self._messaging_infrastructure
                )
                if isinstance(sync_channel_builder, RuntimeChannelBuilderMp):
                    if isinstance(sync_channel_builder.src_process,
                                  RuntimeServiceBuilder):
                        sync_channel_builder.src_process.set_csp_ports(
                            [channel.src_port])
                    else:
                        sync_channel_builder.dst_process.set_csp_ports(
                            [channel.dst_port])
                    if "runtime_to_service" in channel.src_port.name:
                        self.runtime_to_service.append(channel.src_port)
                    elif "service_to_runtime" in channel.src_port.name:
                        self.service_to_runtime.append(channel.dst_port)
                elif isinstance(sync_channel_builder, ServiceChannelBuilderMp):
                    if isinstance(sync_channel_builder.src_process,
                                  RuntimeServiceBuilder):
                        sync_channel_builder.src_process.set_csp_proc_ports(
                            [channel.src_port])
                        self._get_process_builder_for_process(
                            sync_channel_builder.dst_process) \
                            .set_rs_csp_ports([channel.dst_port])
                    else:
                        sync_channel_builder.dst_process.set_csp_proc_ports(
                            [channel.dst_port])
                        self._get_process_builder_for_process(
                            sync_channel_builder.src_process) \
                            .set_rs_csp_ports([channel.src_port])
                else:
                    self.log.info(
                        sync_channel_builder.dst_process.__class__.__name__)
                    raise ValueError("Unexpected type of Sync Channel Builder")

    def _build_processes(self):
        """Builds the process for all process builders within an executable"""
        process_builders: ty.Dict[AbstractProcess, AbstractProcessBuilder] = \
            self._executable.proc_builders
        if process_builders:
            for proc, proc_builder in process_builders.items():
                if isinstance(proc_builder, PyProcessBuilder):
                    # Assign current Runtime to process
                    proc._runtime = self
                    self._messaging_infrastructure.build_actor(target_fn,
                                                               proc_builder)

    def _build_runtime_services(self):
        """Builds the runtime services"""
        runtime_service_builders = self._executable.runtime_service_builders
        if self._executable.runtime_service_builders:
            for _, rs_builder in runtime_service_builders.items():
                self._messaging_infrastructure. \
                    build_actor(target_fn,
                                rs_builder)

    def _get_resp_for_run(self):
        """
        Gets response from RuntimeServices
        """
        if self._is_running:
            for recv_port in self.service_to_runtime:
                data = recv_port.recv()
                if enum_equal(data, MGMT_RESPONSE.REQ_PAUSE):
                    self._req_paused = True
                elif enum_equal(data, MGMT_RESPONSE.REQ_STOP):
                    self._req_stop = True
                elif not enum_equal(data, MGMT_RESPONSE.DONE):
                    if enum_equal(data, MGMT_RESPONSE.ERROR):
                        # Receive all errors from the ProcessModels
                        self._messaging_infrastructure.stop(True)
                        raise RuntimeError(
                            f"Exception(s) occurred. See "
                            f"output above for details.")
                    else:
                        raise RuntimeError(f"Runtime Received {data}")
            if self._req_paused:
                self._req_paused = False
                self.pause()
            if self._req_stop:
                self._req_stop = False
                self.stop()
            self._is_running = False

    def start(self, run_condition: AbstractRunCondition):
        """
        Given a run condition, starts the runtime

        :param run_condition: AbstractRunCondition
        :return: None
        """
        if self._is_initialized:
            # Start running
            self._is_started = True
            self._run(run_condition)
        else:
            self.log.info("Runtime not initialized yet.")

    def _run(self, run_condition: AbstractRunCondition):
        """
        Helper method for starting the runtime

        :param run_condition: AbstractRunCondition
        :return: None
        """
        if self._is_started:
            self._is_running = True
            self._messaging_infrastructure.start()
            if isinstance(run_condition, RunSteps):
                self.num_steps = run_condition.num_steps
                for send_port in self.runtime_to_service:
                    send_port.send(enum_to_np(self.num_steps))
                if run_condition.blocking:
                    self._get_resp_for_run()
            elif isinstance(run_condition, RunContinuous):
                self.num_steps = sys.maxsize
                for send_port in self.runtime_to_service:
                    send_port.send(enum_to_np(self.num_steps))
            else:
                raise ValueError(f"Wrong type of run_condition : "
                                 f"{run_condition.__class__}")
        else:
            self.log.info("Runtime not started yet.")

    def wait(self):
        """Waits for existing run to end. This is helpful if the execution
        was started in non-blocking mode earlier."""
        self._get_resp_for_run()

    def pause(self):
        """Pauses the execution"""
        if self._is_running:
            for send_port in self.runtime_to_service:
                send_port.send(MGMT_COMMAND.PAUSE)
            for recv_port in self.service_to_runtime:
                data = recv_port.recv()
                if not enum_equal(data, MGMT_RESPONSE.PAUSED):
                    if enum_equal(data, MGMT_RESPONSE.ERROR):
                        # Receive all errors from the ProcessModels
                        error_cnt = 0
                        for actors in \
                                self._messaging_infrastructure.actors:
                            actors.join()
                            if actors.exception:
                                _, traceback = actors.exception
                                self.log.info(traceback)
                                error_cnt += 1
                        self.stop()
                        raise RuntimeError(
                            f"{error_cnt} Exception(s) occurred. See "
                            f"output above for details.")
                    else:
                        if recv_port.probe():
                            data = recv_port.recv()
                        if not enum_equal(data, MGMT_RESPONSE.PAUSED):
                            raise RuntimeError(
                                f"{data} Got Wrong Response for Pause.")

            self._is_running = False

    def stop(self):
        """Stops an ongoing or paused run."""
        if self._is_started:
            self.join()
            self._messaging_infrastructure.stop(True)
            self._is_running = False
            self._is_started = False
            # Send messages to RuntimeServices to stop as soon as possible.
        else:
            self.log.info("Runtime not started yet.")

    def join(self):
        """Join all ports and processes"""
        for port in self.runtime_to_service:
            port.join()
        for port in self.service_to_runtime:
            port.join()

    def set_var(self, var_id: int, value: np.ndarray, idx: np.ndarray = None):
        """Sets value of a variable with id 'var_id'."""
        if self._is_running:
            self.log.info(
                "WARNING: Cannot Set a Var when the execution is going on")
            return
        node_config: NodeConfig = self._executable.node_configs[0]

        if var_id not in node_config.var_models:
            self.stop()
            raise AssertionError(
                f"The Var with id <{var_id}> was not associated in the "
                f"ProcModel, thus the current value cannot be "
                f"set.")

        ev: AbstractVarModel = node_config.var_models[var_id]
        runtime_srv_id: int = ev.runtime_srv_id
        model_id: int = ev.proc_id

        if issubclass(list(self._executable.runtime_service_builders.values())
                      [runtime_srv_id].rs_class, AsyncPyRuntimeService):
            raise RuntimeError("Set is not supported in AsyncPyRuntimeService")

        if self._is_started:
            # Send a msg to runtime service given the rs_id that you need value
            # from a model with model_id and var with var_id

            # 1. Send SET Command
            req_port: SendPort = self.runtime_to_service[runtime_srv_id]
            req_port.send(MGMT_COMMAND.SET_DATA)
            req_port.send(enum_to_np(model_id))
            req_port.send(enum_to_np(var_id))

            rsp_port: RecvPort = self.service_to_runtime[runtime_srv_id]

            # 2. Reshape the data
            buffer: np.ndarray = value
            if idx:
                buffer = buffer[idx]
            buffer_shape: ty.Tuple[int, ...] = buffer.shape
            num_items: int = np.prod(buffer_shape).item()
            reshape_order = 'F' if isinstance(ev, LoihiSynapseVarModel) else 'C'
            buffer = buffer.reshape((1, num_items), order=reshape_order)

            # 3. Send [NUM_ITEMS, DATA1, DATA2, ...]
            data_port: SendPort = self.runtime_to_service[runtime_srv_id]
            data_port.send(enum_to_np(num_items))
            for i in range(num_items):
                data_port.send(enum_to_np(buffer[0, i], np.float64))
            rsp = rsp_port.recv()
            if not enum_equal(rsp, MGMT_RESPONSE.SET_COMPLETE):
                raise RuntimeError("Var Set couldn't get successfully "
                                   "completed")
        else:
            raise RuntimeError("Runtime has not started")

    def get_var(self, var_id: int, idx: np.ndarray = None) -> np.ndarray:
        """Gets value of a variable with id 'var_id'."""
        if self._is_running:
            self.log.info(
                "WARNING: Cannot Get a Var when the execution is going on")
            return
        node_config: NodeConfig = self._executable.node_configs[0]

        if var_id not in node_config.var_models:
            self.stop()
            raise AssertionError(
                f"The Var with id <{var_id}> was not associated in the "
                f"ProcModel, thus the current value cannot be "
                f"received.")

        ev: AbstractVarModel = node_config.var_models[var_id]
        runtime_srv_id: int = ev.runtime_srv_id
        model_id: int = ev.proc_id

        if self._is_started:
            # Send a msg to runtime service given the rs_id that you need value
            # from a model with model_id and var with var_id

            # 1. Send GET Command
            req_port: SendPort = self.runtime_to_service[runtime_srv_id]
            req_port.send(MGMT_COMMAND.GET_DATA)
            req_port.send(enum_to_np(model_id))
            req_port.send(enum_to_np(var_id))

            # 2. Receive Data [NUM_ITEMS, DATA1, DATA2, ...]
            data_port: RecvPort = self.service_to_runtime[runtime_srv_id]
            num_items: int = int(data_port.recv()[0].item())
            buffer: np.ndarray = np.empty((1, num_items))
            for i in range(num_items):
                buffer[0, i] = data_port.recv()[0]

            # 3. Reshape result and return
            reshape_order = 'F' if isinstance(ev, LoihiSynapseVarModel) else 'C'
            buffer = buffer.reshape(ev.shape, order=reshape_order)
            if idx:
                return buffer[idx]
            else:
                return buffer
        else:
            raise RuntimeError("Runtime has not started")
