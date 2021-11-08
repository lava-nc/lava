#define PYTHON
#include <python.h>
#include <assert.h>
#include <stdint.h>
#include "ports.h"

size_t port_send(PyObject *port, void *data){ // send via pointer
    PyObject *pyDataObj; // replace with some wrapping of data
    PyObject* n = PyObject_CallMethod(port,"send","O",pyDataObj); // call python port method directly
    assert(PyLong_Check(n));
    return PyLong_AsSize_t((PyLongObject*)n); 
}

size_t port_recv(PyObject *port,void** data){ // recieve pointer to pointer
    PyObject *pyDataObj = PyObject_CallMethod(port,"recv",NULL); // call python port object method
    PyObject *n = PyObject_CallMethod(pyDataObj,"__len__",NULL); // call len(obj)
    *data = (void*) 0; // TODO: replace with assignment of C-format data from pyDataObj
    assert(PyLong_Check(n));
    return PyLong_AsSize_t((PyLongObject*)n); 
}

size_t port_peek(PyObject *port){ // simple call and return of simple type
    PyObject *n = PyObject_CallMethod(port,"peek",NULL);
    assert(PyLong_Check(n));
    return PyLong_AsSize_t((PyLongObject*)n); 
}

size_t port_probe(PyObject *port){ // simple call and return of simple type
    PyObject *n = PyObject_CallMethod(port,"probe",NULL);
    assert(PyLong_Check(n));
    return PyLong_AsSize_t((PyLongObject*)n); 
}
