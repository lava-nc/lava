#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#define PYTHON
#include "methods.h"
#include "run.h"

PyObject* Custom_run(PyObject *self,PyObject* Py_UNUSED(ignored)){
    printf("test_runstate.c run function called\n");
    PyObject *state_port = PyObject_GetAttrString(self, "service_to_process_cmd");
    while(1){
        PyObject *probe_obj = PyObject_CallMethod(state_port,"probe",NULL);
        assert(PyLong_Check(probe_obj));
        size_t probe_size  = PyLong_AsSize_t((PyLongObject*)probe_obj); 
        Py_DECREF(probe_obj);
        if(probe_size){
            PyObject *recv_obj = PyObject_CallMethod(state_port,"recv",NULL);
            assert(PyLong_Check(recv_obj));
            size_t recv_phase  = PyLong_AsSize_t((PyLongObject*)recv_obj); 
            Py_DECREF(recv_obj);
            runState state = {
                .phase = recv_phase,
            };
            if(state.phase){ 
                run(&state);
            } else { // stop if phase is zero
                break;
            }
        }
    }
    Py_DECREF(state_port);
    Py_RETURN_NONE;
}