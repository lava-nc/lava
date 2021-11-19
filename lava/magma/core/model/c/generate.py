import typing as ty
import textwrap as tw


def gen_function(name: str = "MyFunc"):
    return tw.dedent(
        f"""
        static PyObject* Custom_{name}(CustomObject *self,PyObject* Py_UNUSED(ignored)){{
            printf("{name} method called\\n");
            Py_RETURN_NONE;
        }}
        """
    )


def gen_entry(name: str = "MyFunc"):
    return tw.dedent(
        f"""{{{name},(PyCFunction)Custom_{name},METH_NOARGS,"no description"}}"""
    )


def gen_methods(names: ty.List[str]) -> str:
    entries = ",\n            ".join(gen_entry(name) for name in names)
    methods = tw.dedent(
        f"""
        static PyMethodDef Custom_methods[] = {{
            {entries},
            {{NULL}}
            }};
        """
    )
    functions = "".join(gen_function(name) for name in names)
    return tw.dedent(methods + functions)


def gen_port_struct(names: ty.List[str]) -> str:
    fields = "\n            ".join(f"PyObject* {name};" for name in names)
    return tw.dedent(
        f"""
        struct {{
            {fields}
        }} ports;
        """
    )


def gen_ports_load(names: ty.List[str]):
    return "\n".join(
        f"""ports.{name}=PyObject_GetAttrString(self, "{name}");"""
        for name in names
    )


if __name__ == "__main__":
    names = ["a", "b", "c"]
    # print(gen_methods(names))
    print(gen_port_struct(names))
    print(gen_ports_load(names))
