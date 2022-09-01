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
  std::string Name() {
    return name_;
  }
  // pybind11::dtype Dtype() {
  //   return dtype_;
  // }
  // ssize_t* Shape() {
  //   return shape_;
  // }
  size_t Size() {
    return size_;
  }

  int Start() {
    return 0;
  }
  int Probe() {
    return 0;
  }
  int Join() {
    return 0;
  }

 private:
  std::string name_;
  // pybind11::dtype dtype_;
  // ssize_t *shape_ = NULL;
  size_t size_;
  size_t nbytes_;
};

class AbstractSendPort : public AbstractPort {
 public:
  int Send(void* data) {
    return 0;
  }
};

class AbstractRecvPort : public AbstractPort {
 public:
  void* Recv() {
    return NULL;
  }
  int Peek() {
    return 0;
  }
};

using AbstractPortPtr = std::shared_ptr<AbstractPort>;
using AbstractSendPortPtr = std::shared_ptr<AbstractSendPort>;
using AbstractRecvPortPtr = std::shared_ptr<AbstractRecvPort>;

}  // namespace message_infrastructure

#endif  // ABSTRACT_PORT_H_
