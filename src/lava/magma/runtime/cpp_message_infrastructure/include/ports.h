// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORTS_H_
#define PORTS_H_

#include <vector>

#include "abstract_port_implementation.h"

namespace message_infrastrature {

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
    virtual std::vector<> Peek();
    virtual std::vector<> Recv();

    const CppInPortVectorDense  VEC_DENSE;
    const CppInPortVectorSparse VEC_SPARSE;
    const CppInPortScalarDense  SCALAR_DENSE;
    const CppInPortScalarSparse SCALAR_SPARSE;

    AbstractTransformer transformer_;
};

class CppInPortVectorDense : public CppInPort {
  public:
    std::vector<dtype_> Recv();
    std::vector<dtype_> Peek();
};


class CppInPortVectorSparse : public CppInPort {
  public:
    std::vector<dtype_> Recv();
    std::vector<dtype_> Peek();
};


class CppInPortScalarDense : public CppInPort {
  public:
    std::vector<dtype_> Recv();
    std::vector<dtype_> Peek();
};


class CppInPortScalarSparse : public CppInPort {
  public:
    std::vector<dtype_> Recv();
    std::vector<dtype_> Peek();
};


class CppOutPort : public AbstractCppIOPort {
  public:
    virtual std::vector<ndarray> Send();
    virtual void Flush();

    const CppOutPortVectorDense  VEC_DENSE;
    const CppOutPortVectorSparse VEC_SPARSE;
    const CppOutPortScalarDense  SCALAR_DENSE;
    const CppOutPortScalarSparse SCALAR_SPARSE;
};


class CppOutPortVectorDense : public CppOutPort {
  public:
    std::vector<ndarray> Send();
};


class CppOutPortVectorSparse : public CppOutPort {
  public:
    std::vector<ndarray> Send();
};


class CppOutPortScalarDense : public CppOutPort {
  public:
    int Send();
};


class CppOutPortScalarSparse : public CppOutPort {
  public:
    int Send();
};

} // namespace message_infrastrature

#endif