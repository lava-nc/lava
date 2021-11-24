# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
from abc import ABC, abstractmethod, abstractproperty, ABCMeta

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_info

from importlib import import_module, invalidate_caches

import typing as ty
import os
from lava.magma.core.model.interfaces import AbstractPortImplementation

from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.model.c import generate

AbstractCProcessModel = None


class CProcessModelMeta(ABCMeta):
    """
    Self-building python extention class type
    Compiles the sources specified in the class definition and then generates a type that includes the custom module in its base classes
    """

    def __new__(cls, name, bases, attrs):
        if AbstractCProcessModel and AbstractCProcessModel in bases:
            path: str = os.path.dirname(os.path.abspath(__file__))
            sources: ty.List[str] = [
                path + "/" + fname for fname in ["custom.c", "ports_python.c"]
            ]
            methods = generate.get_protocol_methods((cls,) + bases)
            if methods:  # protocol specified
                with open("methods.c", "w") as f:
                    f.write(generate.gen_methods_c(methods))
                sources.append("methods.c")
            else:  # use basic phase run loop
                methods = ["run"]
                sources.append(path + "/run_phases.c")
            with open("methods.h", "w") as f:
                f.write(generate.gen_methods_h(methods))

            def configuration(parent_package="", top_path=None):
                config = Configuration("", parent_package, top_path)
                config.add_extension(
                    "custom",
                    attrs["source_files"] + sources,
                    extra_info=get_info("npymath"),
                )
                return config

            with open(f"proto.h", "w") as f:
                f.write(generate.gen_proto_h(methods))

            if not attrs["source_files"]:
                with open(f"proto.c", "w") as f:
                    f.write(generate.gen_proto_c(methods))
                raise Exception(
                    "no source files given for protocol methods - prototypes automatically generated in proto.c"
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
            module = import_module("custom")
            # from custom import Custom
            bases = (module.Custom,) + bases
            if AbstractPyProcessModel not in bases:
                bases += (AbstractPyProcessModel,)
        return super().__new__(cls, name, bases, attrs)


class AbstractCProcessModel(metaclass=CProcessModelMeta):
    """Abstract interface for a C ProcessModels.

    Example for how variables and ports might be initialized:
        a_in:  LavaCType(CInPort, 'short', precision=16)
        s_out: LavaCType(COutPort, 'short', precision=1)
        u:     LavaCType(array, 'signed long int', precision=24)
        v:     LavaCType(array, 'signed long int, precision=24)
        bias:  LavaCType(array, 'signed short int', precision=12)
        du:    LavaCType(scalar, 'short int', precision=12)
    """

    source_files: ty.List[str] = None
