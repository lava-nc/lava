// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_port.h"

namespace message_infrastructure {

AbstractPort::AbstractPort(const std::string &name, const size_t &size, const size_t &nbytes)
  : name_(name), size_(size), nbytes_(nbytes)
{}

std::string AbstractPort::Name() {
  return this->name_;
}
size_t AbstractPort::Size() {
  return this->size_;
}
}  // namespace message_infrastructure
