// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORTS_H_
#define PORTS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <variant>

#include "abstract_port_implementation.h"
#include "transformer.h"

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

// Forward definition of Vector/Scalar classes
class CppInPortVectorDense;
class CppInPortVectorSparse;
class CppInPortScalarDense;
class CppInPortScalarSparse;


class CppInPort : public AbstractCppIOPort {
 public:
    bool Probe();

    template <typename T>
    T Recv() {}

    template <typename T>
    T Peek() {}

    const CppInPortVectorDense  *kVecDense;
    const CppInPortVectorSparse *kVecSparse;
    const CppInPortScalarDense  *kScalarDense;
    const CppInPortScalarSparse *kScalarSparse;

    // AbstractTransformer transformer_;
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
 public:
    int Recv();
    int Peek();
};


class CppInPortScalarSparse : public CppInPort {
 public:
    std::vector<int> Recv();
    std::vector<int> Peek();
};

// Forward definition of Vector/Scalar classes
class CppOutPortVectorDense;
class CppOutPortVectorSparse;
class CppOutPortScalarDense;
class CppOutPortScalarSparse;

class CppOutPort : public AbstractCppIOPort {
 public:
    template <typename T>
    T Send() {}

    virtual void Flush();

    const CppOutPortVectorDense  *kVecDense;
    const CppOutPortVectorSparse *kVecSparse;
    const CppOutPortScalarDense  *kScalarDense;
    const CppOutPortScalarSparse *KScalarSparse;
};


class CppOutPortVectorDense : public CppOutPort {
 public:
    std::vector<pybind11::dtype> Send();
};


class CppOutPortVectorSparse : public CppOutPort {
 public:
    std::vector<pybind11::dtype> Send();
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

// Forward definition of Vector/Scalar classes
class CppRefPortVectorDense;
class CppRefPortVectorSparse;
class CppRefPortScalarDense;
class CppRefPortScalarSparse;

class CppRefPort : public AbstractCppPort {
 public:
    std::vector<PortPtr> GetPorts();

    template <typename T>
    T Read() {}

    virtual std::vector<pybind11::dtype> Write();
    virtual void Wait();

    const CppRefPortVectorDense  *kVecDense;
    const CppRefPortVectorSparse *kVecSparse;
    const CppRefPortScalarDense  *kScalarDense;
    const CppRefPortScalarSparse *kScalarSparse;
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

// Forward declaration of Vector/Sparse classes
class CppVarPortVectorDense;
class CppVarPortVectorSparse;
class CppVarPortScalarDense;
class CppVarPortScalarSparse;

class CppVarPort : public AbstractCppPort {
 public:
    // std::vector<AbstractCspPort> GetCspPorts();
    virtual void Service();

    template <typename T>
    T Recv() {}

    template <typename T>
    T Peek() {}

    const CppVarPortVectorDense  *kVecDense;
    const CppVarPortVectorSparse *kVecSparse;
    const CppVarPortScalarDense  *kScalarDense;
    const CppVarPortScalarSparse *kScalarSparse;

    // AbstractTransformer transformer_;
    // CspSendPort csp_send_port_;
    // CspRecvPort csp_recv_port_;
    char var_name_[];
};


class CppVarPortVectorDense : public CppVarPort {
 public:
    void Service();
};


class CppVarPortVectorSparse : public CppVarPort {
 public:
    std::vector<pybind11::dtype> Recv();
    std::vector<pybind11::dtype> Peek();
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
