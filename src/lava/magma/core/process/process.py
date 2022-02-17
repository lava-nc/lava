# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
from _collections import OrderedDict
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

# Abbreviation for type annotation in Collection class
mem_type = ty.Union[InPort, OutPort, RefPort, VarPort, Var, "AbstractProcess"]


class Collection:
    """Represents a collection of objects. Member objects can be accessed via
    dot-notation (SomeCollection.some_object_name). Collection also offers an
    iterator to iterate all member objects."""

    def __init__(self, process: "AbstractProcess", name: str):
        """Creates a new Collection.

        :param process: Parent process that holds Collection.
        :param name: Name of Collection.
        """
        self.process: "AbstractProcess" = process
        self.name: str = name
        self._members: ty.Dict[str, mem_type] = OrderedDict()
        self._iterator: int = -1

    @property
    def member_names(self) -> ty.List[str]:
        """Returns the names of Collection members."""
        return list(self._members.keys())

    @property
    def members(self) -> ty.List[mem_type]:
        """Returns the members of the Collection."""
        return list(self._members.values())

    def add_members(self, members: ty.Dict[str, mem_type]):
        """Adds members to Collection.

        :param members: Dictionary of Collection members, where the key is
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


class ProcessPostInitCaller(type):
    """Metaclass for AbstractProcess that overwrites __call__() in order to
    call _post_init() initializer method after __init__() of any sub class
    is called."""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        getattr(obj, "_post_init")()
        return obj


# ToDo: AbstractProcess should inherit from ABC but that throws an error when
#  metaclass is assigned!
class AbstractProcess(metaclass=ProcessPostInitCaller):
    """The notion of a process is inspired by the Communicating Sequential
    Process paradigm for distributed, parallel and asynchronous programming.

    A process represents the fundamental computational unit of the Lava
    framework. Processes are independent from each other in that they primarily
    operate on their own local memory while communication with other
    processes happens via message passing through channels at runtime. This
    makes parallel processing save against side effects from shared-memory
    interaction. Nevertheless, shared-memory interaction between processes is
    also supported. Lava processes are built for executing on a distributed and
    heterogeneous HW platform consisting of different compute resources.
    Compute resources can be conventional CPUs, GPUs, embedded CPUs or
    neuromorphic cores.

    Lava processes consist of the following four key components:
      1. State: A Lava process has internal state that is realized through
      Lava variables or Vars. Vars can be initialized by the user and are
      mutable during execution either as a result of process code or user
      interaction.
      2. Behavior: The behavior of a Lava process is implemented via separate
      so-called ProcessModels. ProcessModels allow to implement the behavior of
      a process at different levels of abstraction, in different languages or
      for different compute resources such as CPUs or neuromorphic cores. A
      single Process can support multiple ProcessModels of the same or
      different type.
      3. Ports: Lava processes communicate with their environment or other
      processes solely via messages. Messages are sent between ports via
      channels. Processes may have one or more input, output or reference
      ports.
      - OutPorts can only be connected to one or moreInPorts and
        communication is uni-directional.
      - InPorts can receive inputs from one or more OutPorts.
      - RefPorts can be connected to Vars of remote processes and allow the
        process having the RefVar to access another process's internal state
        directly and bi-directionally. This type of shared-memory interaction
        is potentially unsafe and should be used with caution!
      4. API: A Lava process can expose a public API to configure or interact
      with a process at setup or during execution interactively. In general,
      the public state Vars can be considered part of a Process's API.

    ProcessModels enable seamless cross-platform execution of processes. In
    particular they allow to build applications or algorithms using processes
    agnostic of the ProcessModel chosen at compile time. There are two
    broad categories of ProcessModels:
    1. LeafProcessModels allow to implement the behavior of a process
    directly in different languages for a particular compute resource.
    ProcessModels specify what Process they implement and what
    SynchronizationProtocol they implement (if necessary for the operation of
    the process). In addition, they specify which compute resource they require
    to function. All this information allows the compiler to map a Process
    and its ProcessModel to the appropriate HW platform.
    2. SubProcessModels allow to implement and compose the behavior of a
    process in terms of other processes. This enables the creation of
    hierarchical processes and reuse of more primitive ProcessModels to
    realize more complex ProcessModels. SubProcessModels inherit all compute
    resource requirements from the sub processes they instantiate. See
    documentation of AbstractProcessModel for more details.

    Developers creating new processes need to inherit from the
    AbstractProcess interface. Any internal Vars or Ports must be initialized
    within its init method:
    ```
    class NewProcess(AbstractProcess):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.var1 = Var(shape=<shape>)
            self.in_port1 = InPort(shape=<shape>)
            self.out_port1 = OutPort(shape=<shape>)
            ...
    ```
    Vars can be initialized with user-provided values and processes can be
    connected to other processes via their ports:
    ```
    p1 = NewProcess(<in_args>)
    p2 = NewProcess(<in_args>)
    p1.out_port1.connect(p2.in_port1)
    ```
    Processes do only specify their states, ports and other public interface
    methods but they do not specify the behavior. Instead, ProcessModels
    specify which Processes they implement. Typically, Processes share their
    states, ports and other public interface methods with their ProcessModels.
    For special cases, one can use proc_params memeber of the process to
    communicate arbitrary object between Processes and their PorcessModels.
    For more information on connecting process, see documentation of InPort,
    OutPort and RefPort.

    Once a concrete Process has been created and connected with other
    processes it needs to be compiled before it is ready for execution.
    A process is compiled by the Lava Compiler while execution is controlled
    by the Lava Runtime. While the Lava Compiler and Runtime can be created
    manually to compile and run a process, the AbstractProcess interface
    provides short-hand methods to compile and run a process without
    interacting with the compiler or runtime directly. In particular running
    an uncompiled process will compile a process automatically.
    Since all processes created in a session usually form a system, calling
    'compile(..)' or 'run(..)' on any of them compiles and runs all of them
    automatically.

    At compile time, the user must provide the Lava compiler with a
    specific instance of a RunConfig class. A RunConfig class represents a set
    of rules that allows the compiler to select one and only one ProcessModel
    of a specific Process to be compiled for execution with specific compute
    resources. See the documentation on RunConfigs for more information how
    to create and customize such RunConfigs.

    Finally, in order to run a process, a RunCondition must be provided. A
    RunCondition such as 'RunSteps' or 'RunContinuous' specifies until when a
    process should be executed.

    Since processes generally run asynchronously and in parallel,
    the execution of a set of processes can either be paused or stopped by
    calling the corresponding 'pause()' or 'stop()' methods.

    In order to save time setting up processes for future use, processes
    can also be saved and reloaded from disk.
    """

    def __init__(self, **kwargs):
        """Initializes the process. Key/value pairs provided by the user will
        used to initialize process interface variables when process is built.
        """
        # Get unique id and parent process from global ProcessServer singleton
        self.id: int = ProcessServer().register(self)

        self.name: str = kwargs.pop("name", f"Process_{self.id}")

        # kwargs will be used for ProcessModel initialization later
        self.init_args: dict = kwargs

        # Containers for InPorts, OutPorts, RefPorts, VarPorts, Vars or SubProcs
        self.out_ports: Collection = Collection(self, "OutPort")
        self.in_ports: Collection = Collection(self, "InPort")
        self.ref_ports: Collection = Collection(self, "RefPort")
        self.var_ports: Collection = Collection(self, "VarPort")
        self.vars: Collection = Collection(self, "Var")
        self.procs: Collection = Collection(self, "SubProcess")

        # Parent process, in case this Process is a sub process of another Proc
        self.parent_proc: ty.Optional[AbstractProcess] = None

        # ProcessModel chosen during compilation
        self._model = None
        self.proc_params = {}

        # Flag indicating whether process has been compiled already
        self._is_compiled: bool = False

        # Current runtime environment
        self._runtime: ty.Optional[Runtime] = None

    def __del__(self):
        """On destruction, terminate Runtime automatically to
        free compute resources.
        """
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
    def is_compiled(self):
        """Returns True if process has been compiled."""
        return self._is_compiled

    @property
    def runtime(self):
        """Returns current Runtime or None if no Runtime exists."""
        return self._runtime

    def register_sub_procs(self, procs: ty.Dict[str, 'AbstractProcess']):
        """Registers other processes as sub processes of this process."""
        for name, p in procs.items():
            if not isinstance(p, AbstractProcess):
                raise AssertionError
            p.parent_proc = self
        self.procs.add_members(procs)

    def validate_var_aliases(self):
        """Validates that any aliased Var is a member of a Process that is a
        strict sub-Process of this Var's Process."""
        for v in self.vars:
            v.validate_alias()

    def is_sub_proc_of(self, proc: 'AbstractProcess'):
        """Returns True, is this Process is a sub process of 'proc'."""
        if self.parent_proc is not None:
            if self.parent_proc == proc:
                return True
            else:
                return self.parent_proc.is_sub_proc_of(proc)
        else:
            return False

    def compile(self, run_cfg: RunConfig) -> Executable:
        """Compiles this and any process connected to this process and
        returns the resulting Executable that can either be serialized or
        passed to Runtime.

        Parameters:
            :param run_cfg: RunConfig is used by compiler to select a
            ProcessModel for each compiled process.
        """
        from lava.magma.compiler.compiler import Compiler
        compiler = Compiler()
        return compiler.compile(self, run_cfg)

    def save(self, path: str):
        """Serializes and saves Process in current stage to disk.
        Serialization at different levels will be possible: After
        partitioning, after mapping, ...
        """
        pass

    def load(self, path: str):
        """Loads and de-serializes Process from disk."""
        pass

    # TODO: (PP) Remove  if condition on blocking as soon as non-blocking
    #  execution is completely implemented
    def run(self,
            condition: AbstractRunCondition = None,
            run_cfg: RunConfig = None):
        """Runs process given RunConfig and RunCondition.

        run(..) compiles this and any process connected to this process
        automatically if it has not been compiled yet.

        run(..) executes for as long as the RunCondition is satisfied when
        RunConfig.blocking == True. Otherwise run(..) returns immediately. In
        this case, wait, pause and stop can be used to interact with the
        Runtime running all processes at a later point:
          - wait(..) blocks execution for as long as the RunCondition is
          satisfied.
          - pause(..) pauses execution at as soon as possible and returns
          control back to the user.
          - stop(..) terminates execution and releases control of all
          involved compute nodes.

        If a run has been suspended by either pause(..) or a RunCondition
        being no longer satisfied, run(..) can be called again to resume
        execution from the current state.

        Parameters:
            :param condition: RunCondition instance specifies for how long to
            run the process.
            :param run_cfg: RunConfig is used by compiler to select a
            ProcessModel for each compiled process.
        """
        if not self._runtime:
            executable = self.compile(run_cfg)
            self._runtime = Runtime(executable,
                                    ActorType.MultiProcessing)
            self._runtime.initialize()

        self._runtime.start(condition)

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
