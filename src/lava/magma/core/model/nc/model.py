# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
from abc import ABC, abstractmethod
import logging
import numpy as np
import typing as ty

from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.nc.ports import AbstractNcPort, NcVarPort
from lava.magma.compiler.channels.pypychannel import CspSelector, CspSendPort, CspRecvPort
from lava.magma.runtime.mgmt_token_enums import (
    MGMT_COMMAND,
    MGMT_RESPONSE,
    enum_equal,
    enum_to_np
)

try:
    from nxsdk.arch.base.nxboard import NxBoard
except(ImportError):
    class NxBoard():
        pass


# ToDo: Move somewhere else. Just created for typing
class AbstractNodeGroup:
    def alloc(self, *args, **kwargs):
        pass


class Net(ABC):
    """Represents a collection of logical entities (Attribute Groups)
    that consume resources on a NeuroCore.

    * InputAxons
    * Synapses
    * DendriticAccumulator
    * Compartments
    * OutputAxons
    * Synaptic pre traces
    * Synaptic post traces
    """
    def __init__(self):
        self.out_ax = AbstractNodeGroup()
        self.cx = AbstractNodeGroup()
        self.cx_profile_cfg = AbstractNodeGroup()
        self.vth_profile_cfg = AbstractNodeGroup()
        self.cx_cfg = AbstractNodeGroup()
        self.da = AbstractNodeGroup()
        self.da_cfg = AbstractNodeGroup()
        self.syn = AbstractNodeGroup()
        self.syn_cfg = AbstractNodeGroup()
        self.in_ax = AbstractNodeGroup()

    def connect(self, from_thing, to_thing):
        pass


class AbstractNcProcessModel(AbstractProcessModel, ABC):
    """Abstract interface for a NeuroCore ProcessModels

    Example for how variables and ports might be initialized:
        a_in: NcInPort =   LavaNcType(NcInPort.VEC_DENSE, float)
        s_out: NcInPort =  LavaNcType(NcOutPort.VEC_DENSE, bool, precision=1)
        u: np.ndarray =    LavaNcType(np.ndarray, np.int32, precision=24)
        v: np.ndarray =    LavaNcType(np.ndarray, np.int32, precision=24)
        bias: np.ndarray = LavaNcType(np.ndarray, np.int16, precision=12)
        du: int =          LavaNcType(int, np.uint16, precision=12)
    """
    def __init__(self, proc_params: ty.Dict[str, ty.Any],
                 loglevel=logging.WARNING) -> None:
        super().__init__(proc_params, loglevel=loglevel)
        self.model_id: ty.Optional[int] = None
        self.service_to_process: ty.Optional[CspRecvPort] = None
        self.process_to_service: ty.Optional[CspSendPort] = None
        self.nc_ports: ty.List[AbstractNcPort] = []
        self.var_ports: ty.List[NcVarPort] = []
        self.var_id_to_var_map: ty.Dict[int, ty.Any] = {}

    def __setattr__(self, key: str, value: ty.Any):
        self.__dict__[key] = value
        if isinstance(value, AbstractNcPort):
            self.nc_ports.append(value)
            # Store all VarPorts for efficient RefPort -> VarPort handling
            if isinstance(value, NcVarPort):
                self.var_ports.append(value)

    @abstractmethod
    def run(self):
        pass

    def join(self):
        self.service_to_process.join()
        self.process_to_service.join()
        for p in self.nc_ports:
            p.join()

    @abstractmethod
    def allocate(self, net: Net):
        """Allocates resources required by Process via Net provided by
        compiler.
        Note: This should work as before.
        """
        pass


class NcProcessModel(AbstractNcProcessModel):
    def __init__(self, proc_params: ty.Dict[str, ty.Any],
                 board: ty.Type[NxBoard],
                 loglevel=logging.WARNING):
        super(AbstractNcProcessModel, self).__init__(proc_params,
                                                     loglevel=loglevel)
        self.board = board

    def start(self):
        self.service_to_process.start()
        self.process_to_service.start()
        for p in self.nc_ports:
            p.start()
        # self.board.start()
        self.run()

    def allocate(self):
        pass

    def run(self):
        """Retrieves commands from the runtime service calls
        their corresponding methods of the ProcessModels. The phase
        is retrieved from runtime service (service_to_process). After
        calling the method of a phase of all ProcessModels the runtime
        service is informed about completion. The loop ends when the
        STOP command is received."""
        selector = CspSelector()
        channel_actions = [(self.service_to_process, lambda: 'cmd')]
        action = 'cmd'
        while True:
            if action == 'cmd':
                cmd = self.service_to_process.recv()
                if enum_equal(cmd, MGMT_COMMAND.STOP):
                    self.board.stop()
                    self.process_to_service.send(MGMT_RESPONSE.TERMINATED)
                    self.join()
                    return
                if enum_equal(cmd, MGMT_COMMAND.PAUSE):
                    self.board.pause()
                    self.process_to_service.send(MGMT_RESPONSE.PAUSED)
                    self.join()
                    return
                try:
                    if enum_equal(cmd, MGMT_COMMAND.RUN):
                        self.process_to_service.send(MGMT_RESPONSE.DONE)
                        num_steps = self.service_to_process.recv()
                        if num_steps > 0:
                            # self.board.run(numSteps=num_steps, aSync=False)
                            self.process_to_service.send(MGMT_RESPONSE.DONE)
                        else:
                            self.log.error(f"Exception: number of time steps"
                                           f"not greater than 0, cannot invoke "
                                           f"run(num_steps) in {self.__class__}")
                            self.process_to_service.send(MGMT_RESPONSE.ERROR)
                    elif enum_equal(cmd, MGMT_COMMAND.GET_DATA):
                        # Handle get/set Var requests from runtime service
                        self._handle_get_var()
                    elif enum_equal(cmd, MGMT_COMMAND.SET_DATA):
                        # Handle get/set Var requests from runtime service
                        self._handle_set_var()
                    else:
                        raise ValueError(
                            f"Wrong Phase Info Received : {cmd}")
                except Exception as inst:
                    self.log.error(f"Exception {inst} occured while"
                                   f" running command {cmd} in {self.__class__}")
                    # Inform runtime service about termination
                    self.process_to_service.send(MGMT_RESPONSE.ERROR)
                    self.join()
                    raise inst
            else:
                # Handle VarPort requests from RefPorts
                self._handle_var_port(action)

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

        # Here get the var from Loihi

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

        # Here set var in Loihi?

    def _handle_var_port(self, var_port):
        """Handles read/write requests on the given VarPort."""
        var_port.service()
