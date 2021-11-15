#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

typedef struct {
    PyObject ob_base; 
    /* Type-specific fields go here. */
    PyArrayObject *x;
} CustomObject;

static void Custom_dealloc(CustomObject* self){
    Py_XDECREF(self->x);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject* Custom_new(PyTypeObject *type,PyObject *args,PyObject *kwds){
    CustomObject *self = (CustomObject*) type->tp_alloc(type,0);if(!self) return NULL; // create self object or fail
    self->x = (PyArrayObject *)PyArray_New(&PyArray_Type,1,(npy_intp[]){1},NPY_INT64,NULL,NULL,0,0,NULL); // create numpy array 
    if(!self->x){Py_DECREF(self);return NULL;} // or destroy and fail
    return (PyObject*) self;
}

static int Custom_init(CustomObject* self,PyObject* args,PyObject* kwargs){
    PyArrayObject *x=NULL,*tmp;
    if(!PyArg_ParseTupleAndKeywords(args,kwargs,"|O",(char*[]){"x",NULL},&x))return -1; // parse or fail
    if(x){
        tmp = self->x; // store old attribute pointer to trash later
        Py_INCREF(x); // take ownership of x
        self->x = x; // assign attribute pointer to new object
        if(tmp) Py_DECREF(tmp); // trash old object
    }
    return 0;//success
}

static PyMemberDef Custom_members[] = {
    {"x",T_OBJECT_EX,offsetof(CustomObject,x),0,"numpy array"},
    {NULL}
};

static PyObject* Custom_run(CustomObject *self,PyObject* Py_UNUSED(ignored)){
    printf("run\n");
    npy_intp n = PyArray_SIZE(self->x);
    for(npy_intp i=0;i<n;++i){
        void* p = PyArray_GETPTR1(self->x,i);
        switch(PyArray_TYPE(self->x)){
            case NPY_INT64:
                printf("int inc\n");
                (*(npy_int64*)p)+=1;
                break;
            case NPY_FLOAT64:
                printf("float inc\n");
                (*(npy_float64*)p)+=1.0f;
                break;
        }
    }
    Py_RETURN_NONE;
}

static PyMethodDef Custom_methods[] = {
    {"inc",(PyCFunction)Custom_run,METH_NOARGS,"x+=1"},
    {NULL}
};

static PyTypeObject CustomType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "custom.Custom",
    .tp_doc = "Custom objects",
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
    .m_name = "custom",
    .m_doc = "module that creates an extension type.",
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
    if (PyModule_AddObject(m, "Custom", (PyObject *) &CustomType) < 0) {
        Py_DECREF(&CustomType);
        Py_DECREF(m);
        return NULL;
    }
    import_array();
    return m;
}