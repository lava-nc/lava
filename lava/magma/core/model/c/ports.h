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
    uint32_t n_csp_ports;
    uint8_t type;
    uint8_t precision;
    uint16_t size;
} Port;

#endif

int port_send(Port *port,void* data);
int port_recv(Port *port,void** data);
int port_peek(Port *port);
int port_probe(Port *port);

#endif

