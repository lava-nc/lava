// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef INCLUDE_ABSTRACT_PORT_IMPLEMENTATION_H_
#define INCLUDE_ABSTRACT_PORT_IMPLEMENTATION_H_

#include <vector>

#include "abstract_port.h"
#include "process_model.h"

namespace message_infrastructure {

class AbstractPortImplementation {
 public:
  int Start();
  int Join();
  std::vector<int> GetShape();
  std::vector<PortPtr> GetPorts();

  DataType dtype_;
  std::vector<int> shape_;
  size_t size_;
  ProcessModel process_model_;
  std::vector<PortPtr> ports_;
};

}  // namespace message_infrastructure

#endif  // INCLUDE_ABSTRACT_PORT_IMPLEMENTATION_H_
