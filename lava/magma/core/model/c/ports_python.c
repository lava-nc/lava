#define PYTHON
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"
#include <assert.h>
#include <stdint.h>
#include "ports.h"

size_t send(PyObject *port, void *data, size_t m){ // send via pointer
    PyObject *pyDataObj = PyArray_New(&PyArray_Type, 1, (npy_intp[]){m},NPY_INT, NULL,data,0, NPY_ARRAY_CARRAY, NULL);
    PyObject* result = PyObject_CallMethod(port,"send","O",pyDataObj); // call python port method directly
    Py_DECREF(pyDataObj);
    assert(PyLong_Check(result));
    size_t n  = PyLong_AsSize_t((PyLongObject*)result); 
    Py_DECREF(result);
    return n;
}

size_t recv(PyObject *port,void** data){ // recieve pointer to pointer
    PyObject *pyDataObj = PyObject_CallMethod(port,"recv",NULL); // call python port object method
    size_t n  = (size_t) PyArray_Size(pyDataObj);
    PyObject *arrayObj = PyArray_ContiguousFromObject(pyDataObj,NPY_INT,0,0);
    PyDECREF(pyDataObj);
    *data = PyArray_DATA(arrayObj); // assign pointer to data pointer
    PyDECREF(pyDataObj); // WARNING: Not sure what to do about this. Possible deallocated pointer or leave in for possible memory leak 
    return n; 
}

size_t peek(PyObject *port){ // simple call and return of simple type
    PyObject *result = PyObject_CallMethod(port,"peek",NULL);
    assert(PyLong_Check(result));
    size_t n  = PyLong_AsSize_t((PyLongObject*)result); 
    Py_DECREF(result);
    return n;
}

size_t probe(PyObject *port){ // simple call and return of simple type
    PyObject *result = PyObject_CallMethod(port,"probe",NULL);
    assert(PyLong_Check(result));
    size_t n  = PyLong_AsSize_t((PyLongObject*)result); 
    Py_DECREF(result);
    return n;
}
