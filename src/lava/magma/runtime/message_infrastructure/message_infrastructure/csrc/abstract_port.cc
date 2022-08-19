// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "abstract_port.h"

namespace message_infrastructure {
std::string AbstractPort::Name() {
  return this->name_;
}
pybind11::dtype AbstractPort::Dtype() {
  return this->dtype_;
}
ssize_t* AbstractPort::Shape() {
  return this->shape_;
}
size_t AbstractPort::Size() {
  return this->size_;
}
}  // namespace message_infrastructure
