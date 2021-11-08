#ifndef _PORTS_H
#define _PORTS_H

#include <stdint.h>

#ifdef PYTHON
#include <Python.h>
typedef PyObject Port;
#else // HOST or EMBEDDED
#include "csp.h"
typedef struct {
    CspPort *csp_ports;
    uint8_t n_csp_ports;
    uint8_t bits;
    uint16_t size;
} Port;

#endif

size_t port_send(Port *port,void* data);
size_t port_recv(Port *port,void** data);
size_t port_peek(Port *port);
size_t port_probe(Port *port);

#endif

