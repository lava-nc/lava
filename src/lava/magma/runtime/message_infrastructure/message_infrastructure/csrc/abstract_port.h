// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_PORT_H_
#define ABSTRACT_PORT_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>
#include <vector>
#include <memory>

#include "shm.h"
#include "utils.h"

namespace message_infrastructure {

class AbstractPort {
 public:
  AbstractPort() {}
  std::string Name() {}
  pybind11::dtype Dtype() {}
  ssize_t* Shape() {}
  size_t Size() {}

  int Start() {}
  int Probe() {}
  int Recv() {}
  int Join() {}

 private:
  std::string name_;
  pybind11::dtype dtype_;
  ssize_t *shape_ = NULL;
  size_t size_;
};

class AbstractSendPort : public AbstractPort {
 public:
  int Send() {}
};

class AbstractRecvPort : public AbstractPort {
 public:
  int Recv() {}
  int Peek() {}
};

using AbstractPortPtr = std::shared_ptr<AbstractPort>;
using AbstractSendPortPtr = std::shared_ptr<AbstractSendPort>;
using AbstractRecvPortPtr = std::shared_ptr<AbstractRecvPort>;

}  // namespace message_infrastructure

#endif  // ABSTRACT_PORT_H_
