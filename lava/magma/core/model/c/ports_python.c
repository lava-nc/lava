#define PYTHON
#include "ports.h"

int port_send(PyObject *port, void *data){ // send via pointer
    PyObject *pyDataObj; // replace with some wrapping of data
    PyObject* status = PyObject_CallMethod(port,"send","O",pyDataObj); // call python port method directly
    return 0; // replace with C value from returned status object
}

int port_recv(PyObject *port,void** data){ // recieve pointer to pointer
    PyObject *pyDataObj = PyObject_CallMethod(port,"recv",NULL);
    *data = (void*) 0; // replace with assignment of C-format data from pyDataObj
    return 0;
}

int port_peek(PyObject *port){ // simple call and return of simple type
    PyObject *result = PyObject_CallMethod(port,"peek",NULL);
    return 0; // replace with result data
}

int port_probe(PyObject *port){ // simple call and return of simple type
    PyObject *result = PyObject_CallMethod(port,"peek",NULL);
    return 0; // replace with result data
}
