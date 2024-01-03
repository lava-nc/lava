# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from abc import ABC, abstractmethod
import logging
from lava.utils.sparse import find
import numpy as np
from scipy.sparse import csr_matrix
import platform

from lava.magma.compiler.channels.pypychannel import (
    CspSendPort,
    CspRecvPort,
    CspSelector,
)
from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.model.py.ports import AbstractPyPort, PyVarPort, PyOutPort
from lava.magma.runtime.mgmt_token_enums import (
    enum_to_np,
    enum_equal,
    MGMT_COMMAND,
    MGMT_RESPONSE,
)
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol


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

    def __init__(
            self,
            proc_params: ty.Type["ProcessParameters"],
            loglevel: ty.Optional[int] = logging.WARNING,
    ) -> None:
        super().__init__(proc_params=proc_params, loglevel=loglevel)
        self.model_id: ty.Optional[int] = None
        self.service_to_process: ty.Optional[CspRecvPort] = None
        self.process_to_service: ty.Optional[CspSendPort] = None
        self.py_ports: ty.List[AbstractPyPort] = []
        self.var_ports: ty.List[PyVarPort] = []
        self.var_id_to_var_map: ty.Dict[int, ty.Any] = {}
        self._selector: CspSelector = CspSelector()
        self._action: str = "cmd"
        self._stopped: bool = False
        self._channel_actions: ty.List[
            ty.Tuple[ty.Union[CspSendPort, CspRecvPort], ty.Callable]
        ] = []
        self._cmd_handlers: ty.Dict[MGMT_COMMAND, ty.Callable] = {
            MGMT_COMMAND.STOP[0]: self._stop,
            MGMT_COMMAND.PAUSE[0]: self._pause,
            MGMT_COMMAND.GET_DATA[0]: self._get_var,
            MGMT_COMMAND.SET_DATA[0]: self._set_var,
        }

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
        if isinstance(value, AbstractPyPort) and value not in self.py_ports:
            self.py_ports.append(value)
            if isinstance(value, PyVarPort) and value not in self.var_ports:
                self.var_ports.append(value)

    def start(self):
        """
        Starts the process model, by spinning up all the ports (mgmt and
        py_ports) and calls the run function.
        """
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
        if isinstance(var, int) or isinstance(var, np.int32):
            data_port.send(enum_to_np(1))
            data_port.send(enum_to_np(var))
        elif isinstance(var, np.ndarray):
            # FIXME: send a whole vector (also runtime_service.py)
            var_iter = np.nditer(var, order="C")
            num_items: np.int32 = np.prod(var.shape)
            data_port.send(enum_to_np(num_items))
            for value in var_iter:
                data_port.send(enum_to_np(value, np.float64))
        elif isinstance(var, csr_matrix):
            _, _, values = find(var, explicit_zeros=True)
            num_items = var.data.size
            data_port.send(enum_to_np(num_items))
            for value in values:
                data_port.send(enum_to_np(value, np.float64))
        elif isinstance(var, str):
            encoded_str = list(var.encode("ascii"))
            data_port.send(enum_to_np(len(encoded_str)))
            for ch in encoded_str:
                data_port.send(enum_to_np(ch, d_type=np.int32))

    def _set_var(self):
        """Handles the set Var command from runtime service."""
        # 1. Receive Var ID and retrieve the Var
        var_id = int(self.service_to_process.recv()[0].item())
        var_name = self.var_id_to_var_map[var_id]
        var = getattr(self, var_name)

        # 2. Receive Var data
        data_port = self.service_to_process
        if isinstance(var, int) or isinstance(var, np.int32):
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
            var_iter = np.nditer(var, op_flags=["readwrite"])
            # Set data one by one
            for i in var_iter:
                if num_items == 0:
                    break
                num_items -= 1
                i[...] = data_port.recv()[0]
            self.process_to_service.send(MGMT_RESPONSE.SET_COMPLETE)
        elif isinstance(var, csr_matrix):
            # First item is number of items
            num_items = int(data_port.recv()[0])

            buffer = np.empty(num_items)
            # Set data one by one
            for i in range(num_items):
                buffer[i] = data_port.recv()[0]
            dst, src, _ = find(var)
            var = csr_matrix((buffer, (dst, src)), var.shape)
            setattr(self, var_name, var)

            self.process_to_service.send(MGMT_RESPONSE.SET_COMPLETE)
        elif isinstance(var, str):
            # First item is number of items
            num_items = int(data_port.recv()[0])

            s = []
            for i in range(num_items):
                s.append(int(data_port.recv()[0]))  # decode string from ascii

            s = bytes(s).decode("ascii")
            setattr(self, var_name, s)
            self.process_to_service.send(MGMT_RESPONSE.SET_COMPLETE)
        else:
            self.process_to_service.send(MGMT_RESPONSE.ERROR)
            raise RuntimeError("Unsupported type")

        # notify PM that Vars have been changed
        self.on_var_update()

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
            if self._action == "cmd":
                cmd = self.service_to_process.recv()[0]
                try:
                    if cmd in self._cmd_handlers:
                        self._cmd_handlers[cmd]()
                        if cmd == MGMT_COMMAND.STOP[0] or self._stopped:
                            return
                    else:
                        raise ValueError(
                            f"Illegal RuntimeService command! ProcessModels of "
                            f"type {self.__class__.__qualname__} "
                            f"{self.model_id} cannot handle "
                            f"command: {cmd} "
                        )
                except Exception as inst:
                    # Inform runtime service about termination
                    self.process_to_service.send(MGMT_RESPONSE.ERROR)
                    self.join()
                    raise inst
            else:
                # Handle VarPort requests from RefPorts
                self._handle_var_port(self._action)
            self._channel_actions = [(self.service_to_process, lambda: "cmd")]
            self.add_ports_for_polling()
            self._action = self._selector.select(*self._channel_actions)

    @abstractmethod
    def add_ports_for_polling(self):
        """
        Add various ports to poll for communication on ports
        """

    def join(self):
        """
        Wait for all the ports to shutdown.
        """
        self.service_to_process.join()
        self.process_to_service.join()
        for p in self.py_ports:
            p.join()

    def on_var_update(self):
        """This method is called if a Var is updated. It
        can be used as callback function to calculate dependent
        changes."""


class PyLoihiProcessModel(AbstractPyProcessModel):
    """
    ProcessModel to simulate a Process on Loihi using CPU.

    The PyLoihiProcessModel implements the same phases of execution as the
    Loihi 1/2 processor but is executed on CPU. See the LoihiProtocol for a
    description of the different phases.

    Example
    -------
    >>> @implements(proc=RingBuffer, protocol=LoihiProtocol)
    >>> @requires(CPU)
    >>> @tag('floating_pt')
    >>> class PySendModel(AbstractPyRingBuffer):
    >>>     # Ring buffer send process model.
    >>>     s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    >>>     data: np.ndarray = LavaPyType(np.ndarray, float)

    >>>     def run_spk(self) -> None:
    >>>         buffer = self.data.shape[-1]
    >>>         self.s_out.send(self.data[..., (self.time_step - 1) % buffer])

    """

    def __init__(self, proc_params: ty.Optional["ProcessParameters"] = None):
        super().__init__(proc_params=proc_params)
        self.time_step: int = 0
        self.phase = PyLoihiProcessModel.Phase.SPK
        self._cmd_handlers.update(
            {
                PyLoihiProcessModel.Phase.SPK[0]: self._spike,
                PyLoihiProcessModel.Phase.PRE_MGMT[0]: self._pre_mgmt,
                PyLoihiProcessModel.Phase.LRN[0]: self._lrn,
                PyLoihiProcessModel.Phase.POST_MGMT[0]: self._post_mgmt,
                PyLoihiProcessModel.Phase.HOST[0]: self._host,
            }
        )
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
        self.advance_to_time_step(self.time_step + 1)
        if self._req_pause or self._req_stop:
            self._handle_pause_or_stop_req()
            return
        if self.lrn_guard() and self.pre_guard():
            self.process_to_service.send(
                PyLoihiProcessModel.Response.REQ_PRE_LRN_MGMT
            )
        elif self.lrn_guard():
            self.process_to_service.send(
                PyLoihiProcessModel.Response.REQ_LEARNING
            )
        elif self.post_guard():
            self.process_to_service.send(
                PyLoihiProcessModel.Response.REQ_POST_LRN_MGMT
            )
        else:
            self.process_to_service.send(
                PyLoihiProcessModel.Response.STATUS_DONE
            )

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
        self.process_to_service.send(PyLoihiProcessModel.Response.REQ_LEARNING)

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
                PyLoihiProcessModel.Response.REQ_POST_LRN_MGMT
            )
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
            PyLoihiProcessModel.Response.STATUS_TERMINATED
        )
        self.join()

    def _pause(self):
        """
        Command handler for Pause Command.
        """
        self.process_to_service.send(PyLoihiProcessModel.Response.STATUS_PAUSED)

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
        if (
                enum_equal(self.phase, PyLoihiProcessModel.Phase.PRE_MGMT)
                or enum_equal(self.phase, PyLoihiProcessModel.Phase.POST_MGMT)
                or enum_equal(self.phase, PyLoihiProcessModel.Phase.HOST)
        ):
            for var_port in self.var_ports:
                for csp_port in var_port.csp_ports:
                    if isinstance(csp_port, CspRecvPort):
                        def func(fvar_port=var_port):
                            return lambda: fvar_port

                        self._channel_actions.insert(
                            0, (csp_port, func(var_port))
                        )

    def advance_to_time_step(self, ts: int):
        """
        Required for output ports which should send a signal to advance time
        """
        for port in self.py_ports:
            if isinstance(port, PyOutPort):
                port.advance_to_time_step(ts)


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

    >>> @implements(proc=SimpleProcess, protocol=AsyncProtocol)
    >>> @requires(CPU)
    >>> class SimpleProcessModel(PyAsyncProcessModel):
    >>>     u = LavaPyType(int, int)
    >>>     v = LavaPyType(int, int)

    >>>     def run_async(self):
    >>>         while True:
    >>>             self.u = self.u + 10
    >>>             self.v = self.v + 1000
    >>>         if self.check_for_stop_cmd():
    >>>             return
    """

    def __init__(self, proc_params: ty.Optional["ProcessParameters"] = None):
        super().__init__(proc_params=proc_params)
        self.num_steps = 0
        self._req_pause = False
        self._req_stop = False
        self._cmd_handlers.update({MGMT_COMMAND.RUN[0]: self._run_async})

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

    def check_for_stop_cmd(self) -> bool:
        """
        Checks if the RS has sent a STOP command.
        """
        if self.service_to_process.probe():
            cmd = self.service_to_process.peek()
            if enum_equal(cmd, MGMT_COMMAND.STOP):
                return True
        return False

    def check_for_pause_cmd(self) -> bool:
        """
        Checks if the RS has sent a PAUSE command.
        """
        if self.service_to_process.probe():
            cmd = self.service_to_process.peek()
            if enum_equal(cmd, MGMT_COMMAND.PAUSE):
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
        self.num_steps = int(self.service_to_process.recv()[0].item())
        self.run_async()
        if self._req_stop:
            self.process_to_service.send(PyAsyncProcessModel.Response.REQ_STOP)
            self._req_stop = False
        elif self._req_pause:
            self.process_to_service.send(PyAsyncProcessModel.Response.REQ_PAUSE)
            self._req_pause = False
        else:
            self.process_to_service.send(
                PyAsyncProcessModel.Response.STATUS_DONE)

    def add_ports_for_polling(self):
        """
        Add various ports to poll for communication on ports
        """
        pass


def _get_attr_dict(
        model_class: ty.Type[PyLoihiProcessModel],
) -> ty.Dict[str, ty.Any]:
    """Get a dictionary of non-callable public attributes of a class.

    Parameters
    ----------
    model_class : ty.Type[PyLoihiProcessModel]
        A class inherited from PyLoihiProcessModel.

    Returns
    -------
    ty.Dict[str, ty.Any]
        Dictionary of attribute name and it's value.
    """
    var_names = [
        v
        for v, m in vars(model_class).items()
        if not (v.startswith("_") or callable(m))
    ]
    var_dict = {
        var_name: getattr(model_class, var_name) for var_name in var_names
    }
    if model_class == PyLoihiProcessModel:
        return {}
    for base in model_class.__bases__:
        var_dict = {**_get_attr_dict(base), **var_dict}
    return var_dict


def _get_callable_dict(
        model_class: ty.Type[PyLoihiProcessModel],
) -> ty.Dict[str, ty.Callable]:
    """Get a dictionary of callable public members of a class.

    Parameters
    ----------
    model_class : ty.Type[PyLoihiProcessModel]
        A class inherited from PyLoihiProcessModel.

    Returns
    -------
    ty.Dict[str, ty.Callable]
        Dictionary of callable name and it's pointer.
    """
    callable_names = [
        v
        for v, m in vars(model_class).items()
        if callable(m) and not v.startswith("_")
    ]
    callable_dict = {
        callable_name: getattr(model_class, callable_name)
        for callable_name in callable_names
    }
    if model_class == PyLoihiProcessModel:
        return {}
    for base in model_class.__bases__:
        callable_dict = {**_get_callable_dict(base), **callable_dict}
    return callable_dict


def PyLoihiModelToPyAsyncModel(
        py_loihi_model: ty.Type[PyLoihiProcessModel],
) -> ty.Type[PyAsyncProcessModel]:
    """Factory function that converts Py-Loihi process models
    to equivalent Py-Async definition.

    Parameters
    ----------
    py_loihi_model : ty.Type[PyLoihiProcessModel]
        Py-Loihi model that describes the functional behavior of a process
        using Loihi Protocol.

    Returns
    -------
    ty.Type[PyAsyncProcessModel]
        Equivalent Py-Async protocol model. The async process model
        class name is the original loihi process model class name with Async
        postfix.
    """
    # The exclude_vars and exclude_callables are
    # based on the constructor of PyLoihiProcessModel and PyAsyncProcModel
    if platform.system() == "Windows":
        raise OSError(
            "Conversion of process models on the fly is not "
            "supported on Windows system. It will result in "
            "pickling error when lava threads for model execution "
            "are spawned. The fundamental reason is Windows OS "
            "does not support forking and needs to use pickle."
        )

    exclude_vars = ["time_step", "phase"]
    exclude_callables = [
        "run_spk",
        "pre_guard",
        "run_pre_mgmt",
        "post_guard",
        "run_post_mgmt",
        "implements_process",
        "implements_protocol",
    ]
    name = py_loihi_model.__name__ + "Async"
    var_dict = _get_attr_dict(py_loihi_model)
    var_dict["implements_process"] = py_loihi_model.implements_process
    var_dict["implements_protocol"] = AsyncProtocol
    callable_dict = {
        k: v
        for k, v in _get_callable_dict(py_loihi_model).items()
        if k not in exclude_callables
    }

    def __init__(self, proc_params: dict):
        # New constructor of the PyAsyncModel implementation.
        PyAsyncProcessModel.__init__(self, proc_params)
        ref_model = py_loihi_model(proc_params)
        attributes = [
            v
            for v, m in vars(ref_model).items()
            if not (v.startswith("_") or callable(m))
            and v not in var_dict.keys()
            and v not in vars(self)
            and v not in exclude_vars
        ]
        for attr in attributes:
            setattr(self, attr, getattr(ref_model, attr))
        self.time_step = 1

    def run_async(self) -> None:
        # New run_async behavior of PyAsyncModel implementation.
        while self.time_step != self.num_steps + 1:
            if py_loihi_model.pre_guard(self):
                py_loihi_model.run_pre_mgmt(self)
            py_loihi_model.run_spk(self)
            if py_loihi_model.post_guard(self):
                py_loihi_model.run_post_mgmt(self)
            self.time_step += 1
            # self.advance_to_time_step(self.time_step)
            for port in self.py_ports:
                if isinstance(port, PyOutPort):
                    port.advance_to_time_step(self.time_step)

    py_async_model = type(
        name,
        (PyAsyncProcessModel,),
        {
            "__init__": __init__,
            "run_async": run_async,
            **var_dict,
            **callable_dict,
        },
    )
    return py_async_model
