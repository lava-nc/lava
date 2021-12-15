/* 
 * Copyright (C) 2021 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * See: https://spdx.org/licenses/
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#define PYTHON
#include "ports.h"
#include "methods.h"
#include "names.h"
typedef struct {
    //PyObject ob_base; 
    PyObject_HEAD
    /* Type-specific fields go here. */
} CustomObject;

static void Custom_dealloc(CustomObject* self){
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject* self_object;

static PyObject* Custom_new(PyTypeObject *type,PyObject *args,PyObject *kwds){
    CustomObject *self = (CustomObject*) type->tp_alloc(type,0);
    if(!self) return NULL;
    self_object = &(self->ob_base);
    return (PyObject*) self;
}

static int Custom_init(CustomObject* self,PyObject* args,PyObject* Py_UNUSED(ignored)){
    return 0;
}

static PyMemberDef Custom_members[] = {
    {NULL}
};

static PyTypeObject CustomType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = FULLNAME ,
    .tp_doc = "Lava Process object",
    .tp_basicsize = sizeof(CustomObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Custom_new,
    .tp_init = (initproc) Custom_init,
    .tp_dealloc = (destructor) Custom_dealloc,
    .tp_members = Custom_members,
    .tp_methods = Custom_methods,
};

static PyModuleDef custommodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = CLASS ,
    .m_doc = "extension module",
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
    if (PyModule_AddObject(m, CLASS , (PyObject *) &CustomType) < 0) {
        Py_DECREF(&CustomType);
        Py_DECREF(m);
        return NULL;
    }
    import_array();
    return m;
}