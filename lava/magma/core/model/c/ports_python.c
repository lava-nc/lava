#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <Python.h>
#include "numpy/arrayobject.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#define PYTHON
#include "ports.h"

extern PyObject* self_object;

void* init_numpy(){
    Py_Initialize;
    import_array();
}


PyObject* get_port(const char* name){
    return PyObject_GetAttrString(self_object, name);
}

size_t send(PyObject *port, void *data, size_t m){ // send via pointer
    init_numpy(); 
    PyObject *pyDataObj = PyArray_New(&PyArray_Type, 1, (npy_intp[]){m},NPY_INT, NULL,data,0, NPY_ARRAY_CARRAY, NULL);
    PyObject* result = PyObject_CallMethod(port,"send","O",pyDataObj); // call python port method directly
    Py_DECREF(pyDataObj);
    assert(PyLong_Check(result));
    size_t n  = PyLong_AsSize_t((PyLongObject*)result); 
    Py_DECREF(result);
    return n;
}

size_t recv(PyObject *port,void** data){ // recieve pointer to pointer
    init_numpy(); 
    PyObject *pyDataObj = PyObject_CallMethod(port,"recv",NULL); // call python port object method
    int dtype = PyArray_DESCR(pyDataObj)->type_num;
    size_t n  = (size_t) PyArray_Size(pyDataObj);
    PyObject *arrayObj = PyArray_ContiguousFromObject(pyDataObj,dtype,0,0);
    void *ptr = PyArray_DATA(arrayObj); // get pointer from array
    *data = ptr; 
    // WARNING: Not sure what to do about this. Possible double-free, deallocated pointer, or memory leak if removed
    // Py_DECREF(arrayObj); 
    // Py_DECREF(pyDataObj); 
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

size_t flush(PyObject *port){ // simple call and return of simple type
    PyObject *result = PyObject_CallMethod(port,"flush",NULL);
    assert(PyLong_Check(result));
    size_t n  = PyLong_AsSize_t((PyLongObject*)result); 
    Py_DECREF(result);
    return n;
}