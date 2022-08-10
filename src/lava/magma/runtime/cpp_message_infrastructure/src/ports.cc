// Copyright (C) 2021-22 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <ports.h>
#include <vector>

namespace message_infrastrature {

std::vector<PortPtr> AbstractCppIOPort::GetPorts() {
  return this->ports_;
}

bool CppInPort::Probe() {
  return;
}


}
