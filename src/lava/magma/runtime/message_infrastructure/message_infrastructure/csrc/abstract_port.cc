// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_port.h"

namespace message_infrastructure {
  std::string AbstractPort::Name() {
    return name_;
  }
  pybind11::dtype AbstractPort::Dtype() {
    return dtype_;
  }
  ssize_t* Shape() {
    return shape_;
  }
  size_t Size() {
    return size_;
  }

}  // namespace message_infrastructure
