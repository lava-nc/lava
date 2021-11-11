#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

typedef struct
{
    PyObject_HEAD
        /* Type-specific fields go here. */
        PyObject *process;
} CustomObject;

static void Custom_dealloc(CustomObject *self)
{
    Py_XDECREF(self->process);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Custom_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    CustomObject *self = (CustomObject *)type->tp_alloc(type, 0);
    if (!self)
        return NULL; // create self object or fail
    return (PyObject *)self;
}

static int Custom_init(CustomObject *self, PyObject *args, PyObject *kwargs)
{
    Py_DECREF(self->process); // TODO: check for okayness
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|", (char *[]){"process", NULL}, &self->process))
        return -1; // parse or fail
    Py_INCREF(self->process);
    return 0; //success
}

static PyMemberDef Custom_members[] = {
    {"process", T_OBJECT_EX, offsetof(CustomObject, process), 0, "state of object as numpy array"},
    {NULL}};

static PyObject *Custom_run(CustomObject *self, PyObject *Py_UNUSED(ignored))
{
    // recieve on this.process.in_port
    PyObject *in_port = PyObject_GetAttrString(self->process, "in_port");
    PyObject *data = PyObject_CallMethod(in_port, "recv", NULL);
    // send on this.process.out_port
    PyObject *out_port = PyObject_GetAttrString(self->process, "out_port");
    PyObject *status = PyObject_CallMethod(out_port, "send", "O", data);
    // clean up python objects
    Py_DECREF(in_port);
    Py_DECREF(data);
    Py_DECREF(out_port);
    Py_DECREF(status);
    // no return value
    Py_RETURN_NONE;
}

static PyMethodDef Custom_methods[] = {
    {"run", (PyCFunction)Custom_run, METH_NOARGS, "do something to process"},
    {NULL}};

static PyTypeObject CustomType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "custom.Custom",
    .tp_doc = "Custom objects",
    .tp_basicsize = sizeof(CustomObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Custom_new,
    .tp_init = (initproc)Custom_init,
    .tp_dealloc = (destructor)Custom_dealloc,
    .tp_members = Custom_members,
    .tp_methods = Custom_methods,
};

static PyModuleDef custommodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "custom",
    .m_doc = "Example module that creates an extension type.",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_custom(void)
{
    PyObject *m;
    if (PyType_Ready(&CustomType) < 0)
        return NULL;

    m = PyModule_Create(&custommodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&CustomType);
    if (PyModule_AddObject(m, "Custom", (PyObject *)&CustomType) < 0)
    {
        Py_DECREF(&CustomType);
        Py_DECREF(m);
        return NULL;
    }
    import_array();
    return m;
}