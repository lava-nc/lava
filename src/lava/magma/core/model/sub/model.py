# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
from abc import abstractmethod

from collections import OrderedDict

from lava.magma.core.model.model import AbstractProcessModel
from lava.magma.core.process.process import AbstractProcess
from dataclasses import is_dataclass, fields


class AbstractSubProcessModel(AbstractProcessModel):
    """Abstract base class for any ProcessModel that derives the behavior of
    the Process it implements from other sub processes.

    Sub classes must implement the __init__ method which must accept the
    Process, that the SubProcessModel implements, as an argument. This allows
    SubProcessModel to access all process attributes such as Vars, Ports or
    initialization arguments passed to the Process constructor via
    proc.init_args.

    Within the ProcessModel constructor, other sub processes can be
    instantiated and connected to each other or the the ports of the
    parent process.
    In addition, Vars of sub processes can be exposed as Vars of the parent
    process by defining an 'alias' relationship between parent process and
    sub process Vars.

    Example:

    >>> class SubProcessModel(AbstractSubProcessModel):
    >>>     def __init__(self, proc: AbstractProcess):
    >>>         # Create one or more sub processes
    >>>         self.proc1 = Proc1(**proc.init_args)
    >>>         self.proc2 = Proc2(**proc.init_args)

    >>>         # Connect one or more ports of sub processes
    >>>         self.proc1.out_ports.out1.connect(self.proc2.in_ports.input1)

    >>>         # Connect one or more ports of parent port with ports of sub
    >>>         # processes
    >>>         proc.in_ports.input1.connect(self.proc1.in_ports.input1)
    >>>         self.proc2.out_ports.output1.connect(proc.out_ports.output1)
    >>>         self.proc1.ref_ports.ref1.connect(proc.ref_ports.ref1)

    >>>         # Define one or more alias relationships between Vars of parent
    >>>         # and sub processes
    >>>         proc.vars.var1.alias(self.proc2.vars.var3)
    """

    @abstractmethod
    def __init__(self, _: AbstractProcess):
        raise NotImplementedError

    def find_sub_procs(self) -> ty.Dict[str, AbstractProcess]:
        """Finds and returns all sub processes of ProcessModel."""
        procs = OrderedDict()
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, AbstractProcess) and \
                    attr is not self.implements_process:
                procs[attr_name] = attr
            if is_dataclass(attr):
                for data in fields(attr):
                    sub_attr = getattr(attr, data.name)
                    if isinstance(sub_attr, AbstractProcess):
                        procs[type(sub_attr).__name__] = sub_attr
        return procs
