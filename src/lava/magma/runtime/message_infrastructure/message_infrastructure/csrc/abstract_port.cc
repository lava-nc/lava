// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_port.h"
#include <string>

namespace message_infrastructure {
std::string AbstractPort::Name() {
  return this->name_;
}
size_t AbstractPort::Size() {
  return this->size_;
}
}  // namespace message_infrastructure
