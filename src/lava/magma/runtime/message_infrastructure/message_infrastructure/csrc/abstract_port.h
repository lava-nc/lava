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
  std::string Name();
  size_t Size();
  int Start();
  int Probe();
  int Join();

  std::string name_;
  size_t size_;
  size_t nbytes_;
};

class AbstractSendPort : public AbstractPort {
 public:
  int Send(MetaDataPtr);
};

class AbstractRecvPort : public AbstractPort {
 public:
  MetaDataPtr Recv();
  int Peek();
};

using AbstractPortPtr = std::shared_ptr<AbstractPort>;
using AbstractSendPortPtr = std::shared_ptr<AbstractSendPort>;
using AbstractRecvPortPtr = std::shared_ptr<AbstractRecvPort>;

}  // namespace message_infrastructure

#endif  // ABSTRACT_PORT_H_
