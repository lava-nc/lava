# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import textwrap as tw

from lava.magma.core.model.interfaces import AbstractPortImplementation


def gen_proto_h(names: ty.List[str]):
    return (
        "#ifndef _PROTO_H\n"
        + "#define _PROTO_H\n"
        + "\n".join(f"int {name}();" for name in names)
        + "\n#endif\n"
    )


def gen_proto_c(names: ty.List[str]) -> str:
    return tw.dedent(
        "".join(
            f"""
            #include "ports.h"

            int {name}(){{
                return 0;
            }}
            """
            for name in names
        )
    )


def get_ports(bases) -> ty.List[str]:
    return [
        v
        for cls in bases
        for v in vars(cls)
        if isinstance(v, AbstractPortImplementation)
    ]


def get_protocol_methods(bases: ty.List[type]) -> ty.List[str]:
    """
    Takes in a list of class bases and returns header and code strings
    """
    return [
        name
        for cls in bases
        for tup in getattr(
            getattr(cls, "implements_protocol", None), "proc_functions", []
        )
        for name in tup
        if name
    ]


def gen_methods(names: ty.List[str]):
    return tw.dedent(
        "".join(
            f"""
        PyObject* Custom_{name}(PyObject *self,PyObject* Py_UNUSED(ignored)){{
            printf("{name} method called\\n");
            return PyLong_FromLong({name}());
        }}
        """
            for name in names
        )
    )


def gen_entry(name: str = "MyFunc"):
    return tw.dedent(
        f"""
        {{
            "{name}",
            (PyCFunction)Custom_{name},
            METH_NOARGS,"no description"
        }}
        """
    )


def gen_method_decls(names: ty.List[str]) -> str:
    return "".join(
        f"PyObject* Custom_{name}(PyObject *self,PyObject* Py_UNUSED(ignored));"
        + "\n"
        for name in names
    )


def gen_method_def(names: ty.List[str]) -> str:
    entries = ",\n            ".join(gen_entry(name) for name in names)
    return tw.dedent(
        f"""
        PyMethodDef Custom_methods[] = {{
            {entries},
            {{NULL}}
            }};
        """
    )


def gen_port_struct(names: ty.List[str]) -> str:
    fields = "\n            ".join(f"PyObject* {name};" for name in names)
    return tw.dedent(
        f"""
        struct {{
            {fields}
        }} ports;
        """
        + "\n"
    )


def gen_ports_load(names: ty.List[str]):
    return "\n".join(
        f"""ports.{name}=PyObject_GetAttrString(self, "{name}");"""
        for name in names
    )


def gen_init(ports: ty.List[str]) -> str:
    return (
        "int Custom_init(CustomObject* self,PyObject* args,"
        + "PyObject* Py_UNUSED(ignored)){\n"
        + tw.indent(gen_ports_load(ports), "    ")
        + "\n    return 0;\n}\n"
    )


def gen_methods_h(methods: ty.List[str]) -> str:
    return (
        "#ifndef _METHODS_H\n"
        + "#define _METHODS_H\n"
        + gen_method_decls(methods)
        + gen_method_def(methods)
        + "#endif\n"
    )
    pass


def gen_methods_c(methods: ty.List[str]) -> str:
    return (
        tw.dedent(
            """
            #define PY_SSIZE_T_CLEAN
            #include <Python.h>
            #include "structmember.h"
            #include <stdio.h>
            #include "proto.h"\n
            """
        )
        + gen_methods(methods)
        + "\n"
    )


def gen_names_h(mod: str, cls: str):
    return tw.dedent(
        f"""
        #ifndef _NAMES_H_
        #define _NAMES_H_

        #define CLASS "{cls}"
        #define MODULE "{mod}"
        #define FULLNAME "{mod}.{cls}"

        #endif
        """
    )


if __name__ == "__main__":
    methods = ["a", "b", "c"]
    ports = ["d", "e", "f"]
    print("=====methods.h======")
    print(gen_methods_h(methods, ports))
    print("=====methods.c======")
    print(gen_methods_c(methods, ports, "mystuff.h"))
    print("=====proto.h========")
    print(gen_proto_h(methods))
    print("=====proto.c========")
    print(gen_proto_c(methods))
    print("==========names.h======")
    print(gen_names_h("Module", "class"))
