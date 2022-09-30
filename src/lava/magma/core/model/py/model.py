# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod
# from functools import partial
import logging
import numpy as np

from message_infrastructure import (SendPort,
                                    RecvPort,
                                    Actor,
                                    ActorStatus,)
from lava.magma.compiler.channels.selector import Selector
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.interfaces import AbstractPortImplementation
from lava.magma.core.model.py.ports import PyVarPort, AbstractPyPort
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

    def __init__(self,
                 proc_params: ty.Type["ProcessParameters"],
                 loglevel: ty.Optional[int] = logging.WARNING) -> None:
        super().__init__(proc_params=proc_params, loglevel=loglevel)
        self.model_id: ty.Optional[int] = None
        self.service_to_process: ty.Optional[RecvPort] = None
        self.process_to_service: ty.Optional[SendPort] = None
        self.py_ports: ty.List[AbstractPortImplementation] = []
        self.var_ports: ty.List[PyVarPort] = []
        self.var_id_to_var_map: ty.Dict[int, ty.Any] = {}
        self._selector: Selector = Selector()
        self._action: str = 'cmd'
        self._stopped: bool = False
        self._actor: Actor = None
        self._channel_actions: ty.List[ty.Tuple[ty.Union[SendPort,
                                                         RecvPort],
                                                ty.Callable]] = []
        self._cmd_handlers: ty.Dict[MGMT_COMMAND, ty.Callable] = {
            MGMT_COMMAND.PAUSE[0]: self._pause,
            MGMT_COMMAND.GET_DATA[0]: self._get_var,
            MGMT_COMMAND.SET_DATA[0]: self._set_var
        }

    def check_status(self):
        actor_status = self._actor.get_status()
        if actor_status in [ActorStatus.StatusStopped,
                            ActorStatus.StatusTerminated,
                            ActorStatus.StatusError]:
            return True
        return False

    def __setattr__(self, key: str, value: ty.Any):
        """
        Sets attribute in the object. This function is used by the builder
        to add ports to py_ports and var_ports list.

        Parameters
        ----------
        key: Attribute being set
        value: Value of the attribute
        -------

        """
        self.__dict__[key] = value
        if isinstance(value, AbstractPyPort):
            self.py_ports.append(value)
            # Store all VarPorts for efficient RefPort -> VarPort handling
            if isinstance(value, PyVarPort):
                self.var_ports.append(value)

    def start(self, actor):
        """
        Starts the process model, by spinning up all the ports (mgmt and
        py_ports) and calls the run function.
        """
        self._actor = actor
        self._actor.set_stop_fn(self._stop)
        self.service_to_process.start()
        self.process_to_service.start()
        for p in self.py_ports:
            p.start()
        self.run()

    def _stop(self):
        """
        Command handler for Stop command.
        """
        self.process_to_service.send(MGMT_RESPONSE.TERMINATED)
        self._stopped = True
        self.join()

    def _pause(self):
        """
        Command handler for Pause command.
        """
        self.process_to_service.send(MGMT_RESPONSE.PAUSED)
        self._actor.status_paused()

    def _get_var(self):
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
            var_iter = np.nditer(var, order='C')
            num_items: np.integer = np.prod(var.shape)
            data_port.send(enum_to_np(num_items))
            for value in var_iter:
                data_port.send(enum_to_np(value, np.float64))

    def _set_var(self):
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
            self.process_to_service.send(MGMT_RESPONSE.SET_COMPLETE)
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
            self.process_to_service.send(MGMT_RESPONSE.SET_COMPLETE)
        else:
            self.process_to_service.send(MGMT_RESPONSE.ERROR)
            raise RuntimeError("Unsupported type")

    def _handle_var_port(self, var_port):
        """Handles read/write requests on the given VarPort."""
        var_port.service()

    def run(self):
        """Retrieves commands from the runtime service and calls their
        corresponding methods of the ProcessModels.
        After calling the method of the ProcessModels, the runtime service
        is informed about completion. The loop ends when the STOP command is
        received."""
        while True:
            if self._action == 'cmd':
                cmd = self.service_to_process.recv()[0]
                try:
                    if cmd in self._cmd_handlers:
                        self._cmd_handlers[cmd]()
                        if cmd == MGMT_COMMAND.STOP[0] or self._stopped:
                            self.join()
                            break
                    else:
                        raise ValueError(
                            f"Illegal RuntimeService command! ProcessModels of "
                            f"type {self.__class__.__qualname__} "
                            f"{self.model_id} cannot handle "
                            f"command: {cmd} ")
                except Exception as inst:
                    self.process_to_service.send(MGMT_RESPONSE.ERROR)
                    self.join()  # join cause raise error
                    self._actor.error()
                    raise inst
            elif self._action is not None:
                self._handle_var_port(self._action)
            self._channel_actions = [(self.service_to_process, lambda: 'cmd')]
            self.add_ports_for_polling()
            self._action = self._selector.select(*self._channel_actions)
            stop = self.check_status()
            if stop:
                self._stop()
                break

    @abstractmethod
    def add_ports_for_polling(self):
        """
        Add various ports to poll for communication on ports
        """
        pass

    def join(self):
        """
        Wait for all the ports to shutdown.
        """
        self.service_to_process.join()
        self.process_to_service.join()
        for p in self.py_ports:
            p.join()
        self._actor.status_terminated()


class PyLoihiProcessModel(AbstractPyProcessModel):
    """
    ProcessModel to simulate a Process on Loihi using CPU.

    The PyLoihiProcessModel implements the same phases of execution as the
    Loihi 1/2 processor but is executed on CPU. See the LoihiProtocol for a
    description of the different phases.

    Example
    =======

        @implements(proc=RingBuffer, protocol=LoihiProtocol)
        @requires(CPU)
        @tag('floating_pt')
        class PySendModel(AbstractPyRingBuffer):
            # Ring buffer send process model.
            s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
            data: np.ndarray = LavaPyType(np.ndarray, float)

            def run_spk(self) -> None:
                buffer = self.data.shape[-1]
                self.s_out.send(self.data[..., (self.time_step - 1) % buffer])

    """

    def __init__(self, proc_params: ty.Optional["ProcessParameters"] = None):
        super().__init__(proc_params=proc_params)
        self.time_step = 0
        self.phase = PyLoihiProcessModel.Phase.SPK
        self._cmd_handlers.update({
            PyLoihiProcessModel.Phase.SPK[0]: self._spike,
            PyLoihiProcessModel.Phase.PRE_MGMT[0]: self._pre_mgmt,
            PyLoihiProcessModel.Phase.LRN[0]: self._lrn,
            PyLoihiProcessModel.Phase.POST_MGMT[0]: self._post_mgmt,
            PyLoihiProcessModel.Phase.HOST[0]: self._host
        })
        self._req_pause: bool = False
        self._req_stop: bool = False

    class Phase:
        """
        Different States of the State Machine of a Loihi Process
        """
        SPK = enum_to_np(1)
        PRE_MGMT = enum_to_np(2)
        LRN = enum_to_np(3)
        POST_MGMT = enum_to_np(4)
        HOST = enum_to_np(5)

    class Response:
        """
        Different types of response for a RuntimeService Request
        """
        STATUS_DONE = enum_to_np(0)
        """Signfies Ack or Finished with the Command"""
        STATUS_TERMINATED = enum_to_np(-1)
        """Signifies Termination"""
        STATUS_ERROR = enum_to_np(-2)
        """Signifies Error raised"""
        STATUS_PAUSED = enum_to_np(-3)
        """Signifies Execution State to be Paused"""
        REQ_PRE_LRN_MGMT = enum_to_np(-4)
        """Signifies Request of PREMPTION before Learning"""
        REQ_LEARNING = enum_to_np(-5)
        """Signifies Request of LEARNING"""
        REQ_POST_LRN_MGMT = enum_to_np(-6)
        """Signifies Request of PREMPTION after Learning"""
        REQ_PAUSE = enum_to_np(-7)
        """Signifies Request of PAUSE"""
        REQ_STOP = enum_to_np(-8)
        """Signifies Request of STOP"""

    def run_spk(self):
        """
        Function that runs in Spiking Phase
        """
        pass

    def run_pre_mgmt(self):
        """
        Function that runs in Pre Lrn Mgmt Phase
        """
        pass

    def run_lrn(self):
        """
        Function that runs in Learning Phase
        """
        pass

    def run_post_mgmt(self):
        """
        Function that runs in Post Lrn Mgmt Phase
        """
        pass

    def pre_guard(self):
        """
        Guard function that determines if pre lrn mgmt phase will get
        executed or not for the current timestep.
        """
        pass

    def lrn_guard(self):
        """
        Guard function that determines if lrn phase will get
        executed or not for the current timestep.
        """
        pass

    def post_guard(self):
        """
        Guard function that determines if post lrn mgmt phase will get
        executed or not for the current timestep.
        """
        pass

    def _spike(self):
        """
        Command handler for Spiking Phase
        """
        self.time_step += 1
        self.phase = PyLoihiProcessModel.Phase.SPK
        self.run_spk()
        if self._req_pause or self._req_stop:
            self._handle_pause_or_stop_req()
            return
        if self.lrn_guard() and self.pre_guard():
            self.process_to_service.send(
                PyLoihiProcessModel.Response.REQ_PRE_LRN_MGMT)
        elif self.lrn_guard():
            self.process_to_service.send(
                PyLoihiProcessModel.Response.REQ_LEARNING)
        elif self.post_guard():
            self.process_to_service.send(
                PyLoihiProcessModel.Response.REQ_POST_LRN_MGMT)
        else:
            self.process_to_service.send(
                PyLoihiProcessModel.Response.STATUS_DONE)

    def _pre_mgmt(self):
        """
        Command handler for Pre Lrn Mgmt Phase
        """
        self.phase = PyLoihiProcessModel.Phase.PRE_MGMT
        if self.pre_guard():
            self.run_pre_mgmt()
        if self._req_pause or self._req_stop:
            self._handle_pause_or_stop_req()
            return
        self.process_to_service.send(
            PyLoihiProcessModel.Response.REQ_LEARNING)

    def _post_mgmt(self):
        """
        Command handler for Post Lrn Mgmt Phase
        """
        self.phase = PyLoihiProcessModel.Phase.POST_MGMT
        if self.post_guard():
            self.run_post_mgmt()
        if self._req_pause or self._req_stop:
            self._handle_pause_or_stop_req()
            return
        self.process_to_service.send(PyLoihiProcessModel.Response.STATUS_DONE)

    def _lrn(self):
        """
        Command handler for Lrn Phase
        """
        self.phase = PyLoihiProcessModel.Phase.LRN
        if self.lrn_guard():
            self.run_lrn()
        if self._req_pause or self._req_stop:
            self._handle_pause_or_stop_req()
            return
        if self.post_guard():
            self.process_to_service.send(
                PyLoihiProcessModel.Response.REQ_POST_LRN_MGMT)
            return
        self.process_to_service.send(PyLoihiProcessModel.Response.STATUS_DONE)

    def _host(self):
        """
        Command handler for Host Phase
        """
        self.phase = PyLoihiProcessModel.Phase.HOST

    def _stop(self):
        """
        Command handler for Stop Command.
        """
        self.process_to_service.send(
            PyLoihiProcessModel.Response.STATUS_TERMINATED)
        self.join()

    def _pause(self):
        """
        Command handler for Pause Command.
        """
        self.process_to_service.send(PyLoihiProcessModel.Response.STATUS_PAUSED)
        self._actor.status_paused()

    def _handle_pause_or_stop_req(self):
        """
        Helper function that checks if stop or pause is being requested by the
        user and handles it.
        """
        if self._req_pause:
            self._req_rs_pause()
        elif self._req_stop:
            self._req_rs_stop()

    def _req_rs_pause(self):
        """
        Helper function that handles pause requested by the user.
        """
        self._req_pause = False
        self.process_to_service.send(PyLoihiProcessModel.Response.REQ_PAUSE)

    def _req_rs_stop(self):
        """
        Helper function that handles stop requested by the user.
        """
        self._req_stop = False
        self.process_to_service.send(PyLoihiProcessModel.Response.REQ_STOP)

    def add_ports_for_polling(self):
        """
        Add various ports to poll for communication on ports
        """
        if enum_equal(self.phase, PyLoihiProcessModel.Phase.PRE_MGMT) or \
                enum_equal(self.phase, PyLoihiProcessModel.Phase.POST_MGMT):
            for var_port in self.var_ports:
                for csp_port in var_port.csp_ports:
                    if isinstance(csp_port, RecvPort):
                        def func(fvar_port=var_port):
                            return lambda: fvar_port
                        self._channel_actions.insert(0,
                                                     (csp_port, func(var_port)))


class PyAsyncProcessModel(AbstractPyProcessModel):
    """
    Process Model for Asynchronous Processes executed on CPU.

    This ProcessModel is used in combination with the AsyncProtocol to
    implement asynchronous execution. This means that the processes could run
    with a varying speed and message passing is possible at any time.

    In order to use the PyAsyncProcessModel, the `run_async()` function
    must be implemented which defines the behavior of the underlying Process.

    Example
    =======

        @implements(proc=SimpleProcess, protocol=AsyncProtocol)
        @requires(CPU)
        class SimpleProcessModel(PyAsyncProcessModel):
            u = LavaPyType(int, int)
            v = LavaPyType(int, int)

            def run_async(self):
                while True:
                    self.u = self.u + 10
                    self.v = self.v + 1000
                    if self.check_for_stop_cmd():
                        return

    """

    def __init__(self, proc_params: ty.Optional["ProcessParameters"] = None):
        super().__init__(proc_params=proc_params)
        self._cmd_handlers.update({
            MGMT_COMMAND.RUN[0]: self._run_async
        })

    class Response:
        """
        Different types of response for a RuntimeService Request
        """
        STATUS_DONE = enum_to_np(0)
        """Signifies Ack or Finished with the Command"""
        STATUS_TERMINATED = enum_to_np(-1)
        """Signifies Termination"""
        STATUS_ERROR = enum_to_np(-2)
        """Signifies Error raised"""
        STATUS_PAUSED = enum_to_np(-3)
        """Signifies Execution State to be Paused"""
        REQ_PAUSE = enum_to_np(-4)
        """Signifies Request of PAUSE"""
        REQ_STOP = enum_to_np(-5)
        """Signifies Request of STOP"""

    def _pause(self):
        """
        Command handler for Pause Command.
        """
        pass

    def _stop(self):
        self._stopped = True
        self.join()

    def check_for_stop_cmd(self) -> bool:
        """
        Checks if the RS has sent a STOP command.
        """
        actor_status = self._actor.get_status()
        if actor_status == ActorStatus.StatusStopped:
            return True
        return False

    def run_async(self):
        """
        User needs to define this function which will run asynchronously when
        RUN command is received.
        """
        raise NotImplementedError("run_async has not been defined")

    def _run_async(self):
        """
        Helper function to wrap run_async function
        """
        self.run_async()
        self.process_to_service.send(PyAsyncProcessModel.Response.STATUS_DONE)

    def add_ports_for_polling(self):
        """
        Add various ports to poll for communication on ports
        """
        pass
