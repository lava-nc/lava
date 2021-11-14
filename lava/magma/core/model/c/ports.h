#ifndef _PORTS_H
#define _PORTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef PYTHON
#include <Python.h>
typedef PyObject Port;
#else 
// HOST or EMBEDDED
#include "csp.h"
typedef struct {
    CspPort *csp_ports;
    uint8_t n_csp_ports;
    uint8_t bits;
    uint16_t size;
} Port;

#endif

size_t send(Port *port,void* data,size_t m);
size_t recv(Port *port,void** data);
size_t peek(Port *port);
size_t probe(Port *port);

#endif

