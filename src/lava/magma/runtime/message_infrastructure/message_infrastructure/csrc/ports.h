// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORTS_H_
#define PORTS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

#include "abstract_port_implementation.h"

namespace message_infrastructure {

class AbstractCppPort : public AbstractPortImplementation {
 public:
    virtual std::vector<PortPtr> GetPorts();

    std::vector<PortPtr> ports_;
};


class AbstractCppIOPort : public AbstractCppPort {
 public:
    std::vector<PortPtr> GetPorts();

    std::vector<PortPtr> ports_;
};


class CppInPort : public AbstractCppIOPort {
 public:
    bool Probe();
    virtual std::vector<pybind11::dtype> Peek();
    virtual std::vector<pybind11::dtype> Recv();

    const CppInPortVectorDense  VEC_DENSE;
    const CppInPortVectorSparse VEC_SPARSE;
    const CppInPortScalarDense  SCALAR_DENSE;
    const CppInPortScalarSparse SCALAR_SPARSE;

    AbstractTransformer transformer_;
};

class CppInPortVectorDense : public CppInPort {
 public:
    std::vector<pybind11::dtype> Recv();
    std::vector<pybind11::dtype> Peek();
};


class CppInPortVectorSparse : public CppInPort {
 public:
    std::vector<pybind11::dtype> Recv();
    std::vector<pybind11::dtype> Peek();
};


class CppInPortScalarDense : public CppInPort {
<<<<<<< HEAD:src/lava/magma/runtime/message_infrastructure/include/ports.h
  public:
    int Recv();
    int Peek();
=======
 public:
    std::vector<pybind11::dtype> Recv();
    std::vector<pybind11::dtype> Peek();
>>>>>>> ff915b6fdba9ecbc1e1c5a2a7f9af2177830db8d:src/lava/magma/runtime/message_infrastructure/message_infrastructure/csrc/ports.h
};


class CppInPortScalarSparse : public CppInPort {
<<<<<<< HEAD:src/lava/magma/runtime/message_infrastructure/include/ports.h
  public:
    std::vector<int> Recv();
    std::vector<int> Peek();
=======
 public:
    std::vector<pybind11::dtype> Recv();
    std::vector<pybind11::dtype> Peek();
>>>>>>> ff915b6fdba9ecbc1e1c5a2a7f9af2177830db8d:src/lava/magma/runtime/message_infrastructure/message_infrastructure/csrc/ports.h
};


class CppOutPort : public AbstractCppIOPort {
 public:
    virtual std::vector<pybind11::array_t<pybind11::dtype>> Send();
    virtual void Flush();

    const CppOutPortVectorDense  VEC_DENSE;
    const CppOutPortVectorSparse VEC_SPARSE;
    const CppOutPortScalarDense  SCALAR_DENSE;
    const CppOutPortScalarSparse SCALAR_SPARSE;
};


class CppOutPortVectorDense : public CppOutPort {
 public:
    std::vector<pybind11::array_t<pybind11::dtype>> Send();
};


class CppOutPortVectorSparse : public CppOutPort {
 public:
    std::vector<pybind11::array_t<pybind11::dtype>> Send();
};


class CppOutPortScalarDense : public CppOutPort {
 public:
    int Send();
};


class CppOutPortScalarSparse : public CppOutPort {
 public:
    int Send();
};

// --------
// RefPorts
// --------
// A CppRefPort is a Port connected to a VarPort of a variable Var of another
// Process. It is used to get or set the value of the referenced Var across
// Processes.
class CppRefPort : public AbstractCppPort {
 public:
    std::vector<PortPtr> GetPorts();
    virtual Read();
    virtual Write();
    virtual Wait();

    const CppRefPortVectorDense  VEC_DENSE;
    const CppRefPortVectorSparse VEC_SPARSE;
    const CppRefPortScalarDense  SCALAR_DENSE;
    const CppRefPortScalarSparse SCALAR_SPARSE;
};


class CppRefPortVectorDense : public CppRefPort {
 public:
    std::vector<pybind11::dtype> Read();
    std::vector<pybind11::dtype> Write();
};


class CppRefPortVectorSparse : public CppRefPort {
 public:
    std::vector<pybind11::dtype> Read();
    std::vector<pybind11::dtype> Write();
};


class CppRefPortScalarDense : public CppRefPort {
 public:
    int Read();
    std::vector<pybind11::dtype> Write();
};


class CppRefPortScalarSparse : public CppRefPort {
 public:
    std::vector<int> Read();
    std::vector<pybind11::dtype> Write();
};

// --------
// VarPorts
// --------
// A CppVarPort is a Port linked to a variable Var of a Process and might be
// connected to a RefPort of another process. It is used to get or set the
// value of the referenced Var across Processes. A CppVarPort is connected via
// two channels to a CppRefPort. One channel is used to send data from the
// CppRefPort to the CppVarPort and the other is used to receive data from the
// CppVarPort. CppVarPort set or send the value of the linked Var (service())
// given the command VarPortCmd received by a connected CppRefPort.
class CppVarPort : public AbstractCppPort {
 public:
    std::vector<AbstractCspPort> GetCspPorts();
    virtual void Service();

    const CppVarPortVectorDense  VEC_DENSE;
    const CppVarPortVectorSparse VEC_SPARSE;
    const CppVarPortScalarDense  SCALAR_DENSE;
    const CppVarPortScalarSparse SCALAR_SPARSE;

    AbstractTransformer transformer_;
    CspSendPort csp_send_port_;
    CspRecvPort csp_recv_port_;
    char var_name_[];
};

class CppVarPortVectorDense : public CppVarPort {
 public:
    void Service();
};

class CppVarPortVectorSparse : public CppVarPort {
 public:
    std::vector<pybind11::array_t<pybind11::dtype>> Recv();
    std::vector<pybind11::array_t<pybind11::dtype>> Peek();
    void Service();
};

class CppVarPortScalarDense : public CppVarPort {
 public:
    int Recv();
    int Peek();
    void Service();
};

class CppVarPortScalarSparse : public CppVarPort {
 public:
    std::vector<int> Recv();
    std::vector<int> Peek();
    void Service();
};

}  // namespace message_infrastructure

#endif  // PORTS_H_
