#define PYTHON
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"
#include <assert.h>
#include <stdint.h>
#include "ports.h"

size_t send(PyObject *port, void *data){ // send via pointer
    PyObject *pyDataObj; // replace with some wrapping of data
    PyObject* result = PyObject_CallMethod(port,"send","O",pyDataObj); // call python port method directly
    assert(PyLong_Check(result));
    size_t n  = PyLong_AsSize_t((PyLongObject*)n); 
    Py_DECREF(result);
    return n;
}

size_t recv(PyObject *port,void** data){ // recieve pointer to pointer
    PyObject *pyDataObj = PyObject_CallMethod(port,"recv",NULL); // call python port object method
    PyObject *result = PyObject_CallMethod(pyDataObj,"__len__",NULL); // call len(obj)
    assert(PyLong_Check(result));
    size_t n  = PyLong_AsSize_t((PyLongObject*)n); 
    Py_DECREF(result);
    PyObject *arrayObj = PyArray_ContiguousFromObject(pyDataObj,NPY_INT,0,0);
    PyDECREF(pyDataObj);
    *data = PyArray_DATA(arrayObj);
    PyDECREF(pyDataObj); // WARNING: Not sure what to do about this. Possible deallocated pointer or leave in for possible memory leak 
    return n; 
}

size_t peek(PyObject *port){ // simple call and return of simple type
    PyObject *result = PyObject_CallMethod(port,"peek",NULL);
    assert(PyLong_Check(result));
    size_t n  = PyLong_AsSize_t((PyLongObject*)n); 
    Py_DECREF(result);
    return n;
}

size_t probe(PyObject *port){ // simple call and return of simple type
    PyObject *result = PyObject_CallMethod(port,"probe",NULL);
    assert(PyLong_Check(result));
    size_t n  = PyLong_AsSize_t((PyLongObject*)n); 
    Py_DECREF(result);
    return n;
}
