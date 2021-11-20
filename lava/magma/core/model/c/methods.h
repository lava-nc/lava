#ifndef _METHODS_H
#define _METHODS_H

// default methods.h for a basic run function

PyObject* Custom_run(PyObject *self,PyObject* Py_UNUSED(ignored));

static PyMethodDef Custom_methods[] = {
    {"run",(PyCFunction)Custom_run,METH_NOARGS,"basic lava run loop"},
    {NULL}
};

#endif