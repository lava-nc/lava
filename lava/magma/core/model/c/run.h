#ifndef _RUN_H
#define _RUN_H

#include <stddef.h>
#include <stdint.h>

/*
// Generate this block in methods.h
static PyObject* Custom_run(PyObject *self,PyObject* Py_UNUSED(ignored));
static PyMethodDef Custom_methods[] = {
    {"run",(PyCFunction)Custom_run,METH_NOARGS,"basic lava run loop"},
    {NULL}
};
*/

typedef struct runState {
    uint8_t phase;
} runState;

void run(runState* state);

#endif