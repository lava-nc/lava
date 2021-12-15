# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from abc import ABCMeta

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_info

from importlib import import_module, invalidate_caches

import typing as ty
import types
import os
import sys

from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.c import generate

AbstractCProcessModel = None


class CProcessModelMeta(ABCMeta):
    """
    Overview:

    Self-building python extention class type
    Compiles the sources specified in the class definition
    and then generates a type that includes the custom module
    in its base classes.

    Run function:

    If no protocol is specified, creates a "run" function that is exposed
    to the python object, which is implemented as a loop that recieves the
    phase from a port named "service_to_process_cmd" and then calls the
    user-defined "run" function with the runState structure. See "run_phase.c".

    Protocols:

    If a protocol is specified, it will assume that there is a "run" function
    implemented in the python class, and exposes the phase functions for the
    run function call. To find the protocol, scans the type heirarchy for
    a protocol, then extracts the method names, and generates a templace
    C code for the user if no code is provided. If code is provided,
    generates a python extension object with pass-through function
    objects that will be called by the protocol, and passed to the user
    code.
    """

    def __new__(cls, name, bases, attrs):
        if AbstractCProcessModel and AbstractCProcessModel in bases:
            path: str = os.path.dirname(os.path.abspath(__file__))
            sources: ty.List[str] = [
                path + "/" + fname for fname in ["custom.c", "ports_python.c"]
            ]
            methods = generate.get_protocol_methods(
                (cls, types.SimpleNamespace(**attrs)) + bases
            )
            if methods:  # protocol specified
                with open("methods.c", "w") as f:
                    f.write(generate.gen_methods_c(methods))
                sources.append("methods.c")
            else:  # use basic phase run loop
                methods = ["run"]
                sources.append(path + "/run_phases.c")
            with open("methods.h", "w") as f:
                f.write(generate.gen_methods_h(methods))
            with open("names.h", "w") as f:
                f.write(generate.gen_names_h("custom", "Custom"))

            def configuration(parent_package="", top_path=None):
                config = Configuration("", parent_package, top_path)
                config.add_extension(
                    "custom",
                    attrs["source_files"] + sources,
                    extra_info=get_info("npymath"),
                )
                return config

            with open("proto.h", "w") as f:
                f.write(generate.gen_proto_h(methods))

            if not attrs["source_files"]:
                with open("proto.c", "w") as f:
                    f.write(generate.gen_proto_c(methods))
                raise Exception(
                    "no source files given for protocol methods"
                    " - prototypes automatically generated in proto.c"
                )

            setup(
                configuration=configuration,
                script_args=[
                    "clean",
                    "--all",
                    "build_ext",
                    "--inplace",
                    "--include-dirs",
                    f"{path}:{os.getcwd()}",
                ],
            )
            invalidate_caches()
            sys.path.append(os.getcwd())
            module = import_module("custom")
            # from custom import Custom
            bases = (module.Custom,) + bases
            if AbstractPyProcessModel not in bases:
                bases += (AbstractPyProcessModel,)
        return super().__new__(cls, name, bases, attrs)


class AbstractCProcessModel(metaclass=CProcessModelMeta):
    """
    To create a process model with behavior in C, inherit from this type and
    provide the source files as an attribute.

    See the documentation on
    CProcessModelMeta for details about compiling and class creation.

    For examples, see: tests/lava/magma/core/model/c/test.py
    """

    source_files: ty.List[str] = None
