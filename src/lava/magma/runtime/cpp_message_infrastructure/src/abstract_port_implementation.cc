// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_port_implementation.h"

int AbstractPortImplementation::Start() {
  for (auto port : this->ports_){
    port->Start();
  }
}

int AbstractPortImplementation::Join() {
  for (auto port : this->ports_){
    port->Join();
  }
}

std::vector<int> AbstractPortImplementation::GetShape() {
  return this->shape_;
}

std::vector<PortPtr> AbstractPortImplementation::GetPorts() {
  return this->ports_;
}