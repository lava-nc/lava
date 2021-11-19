import typing as ty
import textwrap as tw


def gen_proto_h(names: ty.List[str]):
    return (
        "#ifndef _RUN_H\n"
        + "#define _RUN_H\n"
        + "\n".join(f"void {name}();" for name in names)
        + "\n#endif\n"
    )


def gen_proto_c(names: ty.List[str]) -> str:
    return tw.dedent(
        "".join(
            f"""
            void {name}(){{
            }}
            """
            for name in names
        )
    )


def gen_methods(names: ty.List[str]):
    return tw.dedent(
        "".join(
            f"""
        static PyObject* Custom_{name}(CustomObject *self,PyObject* Py_UNUSED(ignored)){{
            printf("{name} method called\\n");
            {name}();
            Py_RETURN_NONE;
        }}
        """
            for name in names
        )
    )


def gen_entry(name: str = "MyFunc"):
    return tw.dedent(
        f"""{{{name},(PyCFunction)Custom_{name},METH_NOARGS,"no description"}}"""
    )


def gen_method_decls(names: ty.List[str]) -> str:
    return "".join(
        f"static PyObject* Custom_{name}(CustomObject *self,PyObject* Py_UNUSED(ignored));"
        + "\n"
        for name in names
    )


def gen_method_def(names: ty.List[str]) -> str:
    entries = ",\n            ".join(gen_entry(name) for name in names)
    return tw.dedent(
        f"""
        static PyMethodDef Custom_methods[] = {{
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
        "static int Custom_init(CustomObject* self,PyObject* args,PyObject* Py_UNUSED(ignored)){\n"
        + tw.indent(gen_ports_load(ports), "    ")
        + "\n    return 0;\n}\n"
    )


def gen_methods_h(methods: ty.List[str], ports: ty.List[str]) -> str:
    return (
        "#ifndef _METHODS_H\n" + "#define _METHODS_H\n"
        "static int Custom_init(CustomObject* self,PyObject* args,PyObject* Py_UNUSED(ignored));\n"
        + gen_method_decls(methods)
        + gen_method_def(methods)
        + gen_port_struct(ports)
        + "#endif\n"
    )
    pass


def gen_methods_c(
    methods: ty.List[str], ports: ty.List[str], header: str
) -> str:
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
        + gen_init(ports)
        + "\n"
        + gen_methods(methods)
        + "\n"
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
