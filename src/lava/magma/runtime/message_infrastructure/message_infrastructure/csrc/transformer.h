// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

#include "abstract_port_implementation.h"

namespace message_infrastructure {

class AbstractTransformer{
 public:
    virtual std::vector<int> Transform(std::vector<int> data);
};

class IdentityTransformer: public AbstractTransformer {
 public:
    std::vector<int> Transform(std::vector<int> data);
};

class VirtualPortTransformer: public AbstractTransformer {
 public:
    std::vector<int> Transform(std::vector<int> data);
    std::vector<int> _Get_Transform();
};

}  // namespace message_infrastructure

#endif  // TRANSFORMER_H_
