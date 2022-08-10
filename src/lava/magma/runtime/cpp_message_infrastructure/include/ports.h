// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORTS_H_
#define PORTS_H_

#include <vector>

#include "abstract_port_implementation.h"

namespace message_infrastructure {

class AbstractCppPort : public AbstractPortImplementation {
  public:
    std::vector<AbstractPort> GetPorts();
};

class AbstractCppIOPort : public AbstractCppPort {
  public:
    std::vector<AbstractPort> GetPorts();
};

class CppInPort : public AbstractCppIOPort {
  public:
    bool Probe();
    void Peek();
    void Recv();

    const CppInPortVectorDense  VEC_DENSE;
    const CppInPortVectorSparse VEC_SPARSE;
    const CppInPortScalarDense  SCALAR_DENSE;
    const CppInPortScalarSparse SCALAR_SPARSE;

    AbstractTransformer transformer_;
};


} // namespace message_infrastructure

#endif