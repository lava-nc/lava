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
    virtual std::vector<pybind11::array_t<pybind11::dtype>>
        Transform(pybind11::array_t<pybind11::dtype> data);
};

class IdentityTransformer: public AbstractTransformer {
 public:
    std::vector<pybind11::array_t<pybind11::dtype>>
        Transform(pybind11::array_t<pybind11::dtype> data);
};

class VirtualPortTransformer: public AbstractTransformer {
 public:
    std::vector<pybind11::array_t<pybind11::dtype>>
        Transform(pybind11::array_t<pybind11::dtype> data);
    std::vector<pybind11::array_t<pybind11::dtype>> _Get_Transform();
};

}  // namespace message_infrastructure

#endif  // TRANSFORMER_H_
