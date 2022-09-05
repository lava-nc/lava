// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_PORT_H_
#define ABSTRACT_PORT_H_

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
  size_t Size() {
    return size_;
  }

  int Start() {
    printf("AbstractPort Start.\n");
    return 0;
  }
  int Probe() {
    printf("AbstractPort Probe.\n");
    return 0;
  }
  int Join() {
    printf("AbstractPort Join.\n");
    return 0;
  }

  std::string name_;
  size_t size_;
  size_t nbytes_;
};

class AbstractSendPort : public AbstractPort {
 public:
  int Send(void* data) {
    printf("AbstractPort Send.\n");
    return 0;
  }
};

class AbstractRecvPort : public AbstractPort {
 public:
  void* Recv() {
    printf("AbstractPort Recv.\n");
    return NULL;
  }
  int Peek() {
    printf("AbstractPort Peek.\n");
    return 0;
  }
};

using AbstractPortPtr = std::shared_ptr<AbstractPort>;
using AbstractSendPortPtr = std::shared_ptr<AbstractSendPort>;
using AbstractRecvPortPtr = std::shared_ptr<AbstractRecvPort>;

}  // namespace message_infrastructure

#endif  // ABSTRACT_PORT_H_
