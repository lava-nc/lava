# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from __future__ import annotations
import logging
import typing as ty
from _collections import OrderedDict
from dataclasses import dataclass
from lava.magma.compiler.executable import Executable

from lava.magma.core.process.interfaces import \
    AbstractProcessMember, IdGeneratorSingleton
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.core.process.ports.ports import \
    InPort, OutPort, RefPort, VarPort
from lava.magma.core.process.variable import Var
from lava.magma.core.run_conditions import AbstractRunCondition
from lava.magma.core.run_configs import RunConfig
from lava.magma.runtime.runtime import Runtime

if ty.TYPE_CHECKING:
    from lava.magma.core.model.model import AbstractProcessModel


class ProcessPostInitCaller(type):
    """Metaclass for AbstractProcess that overwrites __call__() in order to
    call _post_init() initializer method after __init__() of any sub class
    is called.
    """
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        getattr(obj, "_post_init")()
        return obj


class AbstractProcess(metaclass=ProcessPostInitCaller):
    """The notion of a Process is inspired by the Communicating Sequential
    Process paradigm for distributed, parallel, and asynchronous programming.

    A Process represents the fundamental computational unit of the Lava
    framework. Processes are independent from each other in that they primarily
    operate on their own local memory while communication with other
    Processes happens via message passing through channels at runtime. This
    makes parallel processing safe against side effects from shared-memory
    interaction. Nevertheless, shared-memory interaction between Processes is
    also supported. Lava Processes are built for executing on a distributed and
    heterogeneous hardware platform consisting of different compute resources.
    Compute resources can be conventional CPUs, GPUs, embedded CPUs, or
    neuromorphic cores.

    Lava Processes consist of the following key components:

    1. State: A Lava Process has internal state that is realized through
       Lava Vars (variables). Vars can be initialized by the user
       and are mutable during execution either as a result of Process code or
       user interaction.
    2. Ports: Lava Processes communicate with their environment or other
       Processes solely via messages. Messages are sent between ports via
       channels. Processes may have one or more input, output, or reference
       ports.

       - OutPorts can only be connected to one or more InPorts and
         communication is uni-directional.
       - InPorts can receive inputs from one or more OutPorts.
       - RefPorts can be connected to Vars of remote Processes and allow the
         Process with the RefPort to access another Process' internal state
         directly and bi-directionally. This type of shared-memory interaction
         is potentially unsafe and should be used with caution!

    3. API: A Lava Process can expose a public API to configure it or interact
       with it at setup or during execution interactively. In general,
       the public state Vars can be considered part of a Process' API.

    Crucially, the behavior of a Process is not specified by the Process itself
    but by separate ProcessModels that implement the behavior
    of a Process at different levels of abstraction, in different
    programming languages, and for different compute resources, such as
    CPUs or neuromorphic cores. A single Process can support multiple
    ProcessModels of the same type or of different types. Please refer to
    the documentation of AbstractProcessModel for more details.

    Developers creating new Processes need to inherit from the
    AbstractProcess interface. Any internal Vars or Ports must be initialized
    within its init method:

    >>> class NewProcess(AbstractProcess):
    >>>     def __init__(self, shape, name):
    >>>         super().__init__(shape=shape, name=name)
    >>>         self.var1 = Var(shape=shape)
    >>>         self.in_port1 = InPort(shape=shape)
    >>>         self.out_port1 = OutPort(shape=shape)
    >>>         ...

    Vars should only be used for dynamic state parameters that play a role in
    the behavioral model of the Process or for static configuration parameters
    that affect the behavior of the Process (e.g., the membrane potential of a
    LIF neuron).
    Meta parameters that are only needed for communicating information between
    the Process and its ProcessModels (e.g., shape) should not become a Var.
    They should be passed to the Process as keyword arguments and
    then need to be passed to the __init__ method of AbstractProcess,
    as is done with 'shape' and 'name' in the example above. All such keyword
    arguments are stored in the member 'proc_params', which is passed on to all
    ProcessModels of the Process.

    Vars can be initialized with user-provided values and Processes can be
    connected to other Processes via their ports:
    ```
    p1 = NewProcess(<in_args>)
    p2 = NewProcess(<in_args>)
    p1.out_port1.connect(p2.in_port1)
    ```
    For more information on connecting Processes, see the documentation of
    InPort, OutPort, and RefPort.

    Once a concrete Process has been created and connected with other
    Processes it needs to be compiled before it is ready for execution.
    A Process is compiled by the Lava Compiler while execution is controlled
    by the Lava Runtime. While the Lava Compiler and Runtime can be created
    manually to compile and run a Process, the AbstractProcess interface
    provides short-hand methods to compile and run a Process without
    interacting with the compiler or runtime directly. In particular running
    an uncompiled Process will compile a Process automatically.
    Since all Processes created in a session usually form a system, calling
    'compile(..)' or 'run(..)' on any of them compiles and runs all of them
    automatically.

    At compile time, the user must provide the Lava Compiler with a
    specific instance of a RunConfig class. A RunConfig class represents a set
    of rules that allows the compiler to select one and only one ProcessModel
    of a specific Process to be compiled for execution with specific compute
    resources. See the documentation on RunConfigs for more information how
    to create and customize such RunConfigs.

    Finally, in order to run a Process, a RunCondition must be provided. A
    RunCondition such as 'RunSteps' or 'RunContinuous' specifies until when a
    Process should be executed.

    Since Processes generally run asynchronously and in parallel,
    the execution of a set of Processes can either be paused or stopped by
    calling the corresponding 'pause()' or 'stop()' methods.

    In order to save time setting up Processes for future use, Processes
    can also be saved and reloaded from disk.

    Parameters
    ----------
    proc_params
        Any keyword arguments that get passed from child
        Processes will be stored in the AbstractProcess member
        'proc_params' and passed to all ProcessModels. 'proc_params' can
        be further added to or modified in order to pass parameters
        to ProcessModels that are not represented by dynamic state
        variables (Vars) of the Process.
    name : str, optional
        Name of the Process. Default is 'Process_ID', where ID
        is an integer value that is determined automatically.
    log_config : LogConfig, optional
        Configuration options for logging.
    """

    def __init__(self, **proc_params) -> None:
        """Initializes a new Process."""
        # Keyword arguments passed to the Process child class. This is saved
        # for use in ProcessModels and can be further populated here.
        self.proc_params = ProcessParameters(initial_parameters=proc_params)

        # Get unique id and parent process from global ProcessServer singleton.
        self.id: int = ProcessServer().register(self)

        self.name: str = proc_params.get("name") or f"Process_{self.id}"

        # Setup Logging
        self.log = logging.getLogger()
        self._log_config = proc_params.get("log_config") or LogConfig(
            file="lava.log")
        formatter = logging.Formatter(self._log_config.format,
                                      datefmt=self._log_config.date_format)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._log_config.level_console)
        console_handler.setFormatter(formatter)
        self.log.addHandler(console_handler)

        if self._log_config.logs_to_file:
            logging.basicConfig(
                filename=self._log_config.file,
                level=self._log_config.level,
                format=self._log_config.format,
                datefmt=self._log_config.date_format
            )

        # Containers for InPorts, OutPorts, RefPorts, VarPorts, Vars,
        # or SubProcs.
        self.out_ports: Collection = Collection(self, "OutPort")
        self.in_ports: Collection = Collection(self, "InPort")
        self.ref_ports: Collection = Collection(self, "RefPort")
        self.var_ports: Collection = Collection(self, "VarPort")
        self.vars: Collection = Collection(self, "Var")
        self.procs: Collection = Collection(self, "SubProcess")

        # Parent process, in case this Process is a sub process of another Proc.
        self.parent_proc: ty.Optional[AbstractProcess] = None

        # ProcessModel chosen during compilation
        self._model: ty.Optional["AbstractProcessModel"] = None

        # ProcessModel class chosen during compilation
        self._model_class = None

        # Flag indicating whether process has been compiled already.
        self._is_compiled: bool = False

        # folded view
        self._folded_view : ty.Optional[AbstractProcess] = None
        self._folded_view_inst_id : int = -1

        # Current runtime environment
        self._runtime: ty.Optional[Runtime] = None

    def __del__(self):
        """On destruction, terminate Runtime automatically to
        free compute resources.
        """
        self.stop()

    def __enter__(self):
        """Executed when Process enters a "with" block of a context manager."""

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the runtime when exiting "with" block of a context manager."""
        self.stop()

    def _post_init(self):
        """Called after __init__() method of any sub class via
        ProcessMetaClass to finalize initialization leading to following
        behavior in subclass:
            def __init__(self, ..):
                super(AbstractProcess, self).__init__(..)
                # Custom subclass initialization code
                _post_init()
        """
        attrs = self._find_attr_by_type(OutPort)
        self._init_proc_member_obj(attrs)
        self.out_ports.add_members(attrs)
        attrs = self._find_attr_by_type(InPort)
        self._init_proc_member_obj(attrs)
        self.in_ports.add_members(attrs)
        attrs = self._find_attr_by_type(RefPort)
        self._init_proc_member_obj(attrs)
        self.ref_ports.add_members(attrs)
        attrs = self._find_attr_by_type(VarPort)
        self._init_proc_member_obj(attrs)
        self.var_ports.add_members(attrs)

        attrs = self._find_attr_by_type(Var)
        self._init_proc_member_obj(attrs)
        self.vars.add_members(attrs)

        attrs = self._find_attr_by_type(AbstractProcess)
        self._init_proc_member_obj(attrs)
        self.procs.add_members(attrs)

    def _find_attr_by_type(self, cls) -> ty.Dict:
        """Finds all class attributes of a certain class type."""
        attrs = dict()
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, cls):
                attrs[attr_name] = attr
        return attrs

    def _init_proc_member_obj(self, attrs: ty.Dict[str, AbstractProcessMember]):
        """Initializes AbstractProcessMember by assigning them names and their
        parent process."""
        for attr_name, attr in attrs.items():
            attr.name = attr_name
            attr.process = self

    @property
    def runtime(self):
        """Returns current Runtime or None if no Runtime exists."""
        return self._runtime

    @runtime.setter
    def runtime(self, value):
        """Returns current Runtime or None if no Runtime exists."""
        self._runtime = value

    def register_sub_procs(self, procs: ty.Dict[str, AbstractProcess]):
        """Registers other processes as sub processes of this process."""
        for p in procs.values():
            if not isinstance(p, AbstractProcess):
                raise AssertionError
            p.parent_proc = self
        self.procs.add_members(procs)

    def validate_var_aliases(self):
        """Validates that any aliased Var is a member of a Process that is a
        strict sub-Process of this Var's Process."""
        for v in self.vars:
            v.validate_alias()

    def is_sub_proc_of(self, proc: AbstractProcess):
        """Returns True, is this Process is a sub process of 'proc'."""
        if self.parent_proc is not None:
            if self.parent_proc == proc:
                return True
            else:
                return self.parent_proc.is_sub_proc_of(proc)
        else:
            return False

    def propagate_folded_views(self):
        for p in self.procs:
            p._folded_view = self._folded_view
            p._folded_view_inst_id = self._folded_view_inst_id
            p.propagate_folded_views()

    def run(self,
            condition: AbstractRunCondition,
            run_cfg: ty.Optional[RunConfig] = None,
            compile_config: ty.Optional[ty.Dict[str, ty.Any]] = None):
        """Executes this and any connected Processes that form a Process
        network. If any Process has not been compiled, it is automatically
        compiled before execution.

        When condition.blocking is True, the Processes are executed for as long
        as the RunCondition is satisfied. Otherwise, the method run() returns
        immediately while the Processes are executed. In this case,
        the methods wait(), pause(), and stop() can be used to
        interact with the Runtime:

        - wait() blocks execution for as long as the RunCondition is
          satisfied.
        - pause() pauses execution as soon as possible and returns
          control back to the user.
        - stop() terminates execution and releases control of all
          involved compute nodes.

        If a run has been suspended by either pause() or a RunCondition
        being no longer satisfied, run() can be called again to resume
        execution from the current state.

        NOTE: run_cfg will be ignored when re-running a previously compiled
        process.

        Parameters
        ----------
        condition : AbstractRunCondition
            Specifies for how long to run the Process.
        run_cfg : RunConfig, optional
            Used by the compiler to select a ProcessModel for each Process.
            Must be provided when Processes have to be compiled, can be
            omitted otherwise.
        compile_config: Dict[str, Any], optional
            Configuration options for the Compiler and SubCompilers.
        """
        if not self._runtime:
            if not run_cfg:
                raise ValueError("run_cfg must not be None when calling"
                                 " Process.run() unless the process has already"
                                 " been compiled.")
            self.create_runtime(run_cfg=run_cfg, compile_config=compile_config)
        self._runtime.start(condition)

    def create_runtime(self, run_cfg: ty.Optional[RunConfig] = None,
                       executable: ty.Optional[Executable] = None,
                       compile_config:
                       ty.Optional[ty.Dict[str, ty.Any]] = None):
        """Creates a runtime for this process and all connected processes by
        compiling the process to an executable and assigning that executable to
        the process and connected processes.

        See Process.run() for information on Process blocking, which must be
        specified in the run_cfg passed to create_runtime.

        Parameters
        ----------
        run_cfg : RunConfig, optional
            Used by the compiler to select a ProcessModel for each Process.
            Must be provided when Processes have to be compiled, can be
            omitted otherwise.
        compile_config: Dict[str, Any], optional
            Configuration options for the Compiler and SubCompilers.
        """
        if executable is None:
            executable = self.compile(run_cfg, compile_config)
        self._runtime = Runtime(executable,
                                ActorType.MultiProcessing,
                                loglevel=self._log_config.level)
        executable.assign_runtime_to_all_processes(self._runtime)
        self._runtime.initialize()

    def compile(self,
                run_cfg: RunConfig,
                compile_config: ty.Optional[ty.Dict[str, ty.Any]] = None
                ) -> Executable:
        """Compiles this and any process connected to this process and
        returns the resulting Executable that can either be serialized or
        passed to Runtime.

        Parameters
        ----------
        run_cfg : RunConfig
            Used by the compiler to select a ProcessModel for each Process.
        compile_config: Dict[str, Any], optional
            Configuration options for the Compiler and SubCompilers.
        """
        from lava.magma.compiler.compiler import Compiler
        compiler = Compiler(compile_config, self._log_config.level)
        return compiler.compile(self, run_cfg)

    def wait(self):
        """Waits until end of process execution or for as long as
        RunCondition is met by blocking execution at command line level."""
        if self.runtime:
            self.runtime.wait()

    def pause(self):
        """Pauses process execution while running in non-blocking mode."""
        if self.runtime:
            self.runtime.pause()

    def stop(self):
        """Terminates process execution by releasing all allocated compute
        nodes."""
        if self.runtime:
            self.runtime.stop()

    def save(self, path: str):
        """Serializes and saves Process in current stage to disk.
        Serialization at different levels will be possible: After
        partitioning, after mapping, ...
        """
        raise NotImplementedError

    def load(self, path: str):
        """Loads and de-serializes Process from disk."""
        raise NotImplementedError

    @property
    def model(self) -> "AbstractProcessModel":
        """ Return model """
        return self._model

    @model.setter
    def model(self, process_model: "AbstractProcessModel"):
        self._model = process_model

    @property
    def model_class(self) -> ty.Type["AbstractProcessModel"]:
        """ Return model class """
        return self._model_class

    @property
    def is_compiled(self):
        """Returns True if process has been compiled."""
        return self._is_compiled

    @property
    def folded_view(self) -> ty.Type["AbstractProcess"]:
        """ Return folded view process"""
        return self._folded_view

    @folded_view.setter
    def folded_view(self, folded_view : ty.Optional[AbstractProcess] = None):
        self._folded_view = folded_view

    @property
    def folded_view_inst_id(self) -> int:
        return self._folded_view_inst_id

    @folded_view_inst_id.setter
    def folded_view_inst_id(self, inst_id : int = -1):
        self._folded_view_inst_id = inst_id


class ProcessParameters:
    """Wrapper around a dictionary that is used to pass parameters from a
    Process to its ProcessModels. The dictionary can be filled with an
    initial dictionary of parameters. Any further changes via the __setitem__
    method may not overwrite existing values. To overwrite a value, use the
    method overwrite().

    Parameters
    ----------
    initial_parameters : Dict[str, Any]
        Initial dictionary of parameters for a Process/ProcessModel.
    """
    def __init__(self, initial_parameters: ty.Dict[str, ty.Any]) -> None:
        self._parameters = initial_parameters

    def __setitem__(self, key: str, value: ty.Any) -> None:
        self._assert_key_is_unknown(key)
        self._parameters[key] = value

    def _assert_key_is_unknown(self, key) -> None:
        if key in self._parameters:
            raise KeyError

    def overwrite(self, key, value) -> None:
        """Sets a key-value pair without checking whether the key is already
        present in ProcessParameters."""
        self._parameters[key] = value

    def __getitem__(self, key: str) -> ty.Any:
        return self._parameters[key]

    def get(self, key, default=None):
        if key not in self._parameters:
            return default
        else:
            return self._parameters[key]


@dataclass
class LogConfig:
    """Configuration options for logging that can be passed into a Process."""
    file: str = ""
    level: int = logging.WARNING
    level_console: int = logging.ERROR
    logs_to_file: bool = False
    format: str = "%(asctime)s:%(levelname)s: %(name)s - %(message)s"
    date_format: str = "%m/%d/%Y %I:%M:%S%p"

    def __post_init__(self) -> None:
        if self.logs_to_file and self.file == "":
            raise ValueError("Please provide a file name to log to when "
                             "setting logs_to_file=True.")


class ProcessServer(IdGeneratorSingleton):
    """ProcessServer singleton keeps track of all existing processes and issues
    new globally unique process ids."""

    instance: ty.Optional["ProcessServer"] = None
    is_not_initialized: bool = True

    def __new__(cls):
        if ProcessServer.instance is None:
            ProcessServer.instance = object.__new__(ProcessServer)
        return ProcessServer.instance

    def __init__(self):
        if ProcessServer.is_not_initialized:
            super().__init__()
            self.processes: ty.List["AbstractProcess"] = []
            ProcessServer.is_not_initialized = False

    @property
    def num_processes(self):
        """Returns number of processes created so far."""
        return len(self.processes)

    def register(self, process: AbstractProcess) -> int:
        """Registers a process with ProcessServer."""
        if not isinstance(process, AbstractProcess):
            raise AssertionError("'process' must be an AbstractProcess.")
        self.processes.append(process)
        return self.get_next_id()

    def reset_server(self):
        """Resets the ProcessServer to initial state."""
        self.processes = []
        self._next_id = 0
        ProcessServer.reset_singleton()


class Collection:
    """Represents a collection of objects. Member objects can be accessed via
    dot-notation (SomeCollection.some_object_name). Collection also offers an
    iterator to iterate all member objects.

    Parameters
    ----------
    process : AbstractProcess
        Parent Process that holds the Collection
    name : str
        Name of the Collection
    """
    # Abbreviation for type annotation in Collection class
    mem_type = ty.Union[
        InPort, OutPort, RefPort, VarPort, Var, "AbstractProcess"]

    def __init__(self, process: AbstractProcess, name: str) -> None:
        """Creates a new Collection."""
        self.process: "AbstractProcess" = process
        self.name: str = name
        self._members: ty.Dict[str, Collection.mem_type] = OrderedDict()
        self._iterator: int = -1
        self._is_coroutine = False

    @property
    def member_names(self) -> ty.List[str]:
        """Returns the names of Collection members."""
        return list(self._members.keys())

    @property
    def members(self) -> ty.List[mem_type]:
        """Returns the members of the Collection."""
        return list(self._members.values())

    def add_members(self, members: ty.Dict[str, mem_type]):
        """Adds members to a Collection.

        Parameters
        ----------
        members : Dict[str, mem_type]
            Dictionary of Collection members, where the key is
            the string name of the member and the value is the member.
        """
        self._members.update(members)
        for key, mem in members.items():
            setattr(self, key, mem)

    def is_empty(self) -> bool:
        """Returns True if Collection has no members."""
        return len(self.members) == 0

    def has(self, obj) -> bool:
        """Returns True if member is in collection."""
        if not hasattr(obj, "name"):
            raise AssertionError
        return obj.name in self.member_names

    def __getattr__(self, item):
        if not isinstance(item, str):
            raise AssertionError

        if item not in self.member_names:
            raise AssertionError(
                f"'{item}' is not a member of '{self.name}' collection of "
                f"process '"
                f"{self.process.name}::{self.process.__class__.__name__}'"
            )
        return getattr(self, item)

    def __iter__(self):
        return self

    def __next__(self):
        self._iterator += 1
        if self._iterator < len(self.members):
            return getattr(self, self.member_names[self._iterator])
        self._iterator = -1
        raise StopIteration

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
