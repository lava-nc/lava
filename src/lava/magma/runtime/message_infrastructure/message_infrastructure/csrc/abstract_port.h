// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_PORT_H_
#define ABSTRACT_PORT_H_

#include <string>
#include <vector>
#include <memory>

#include "utils.h"

namespace message_infrastructure {

class AbstractPort {
 public:
  AbstractPort(const std::string &name, const size_t &size,
    const size_t &nbytes);
  AbstractPort() = default;
  virtual ~AbstractPort() = default;

  std::string Name();
  size_t Size();
  virtual void Start() = 0;
  virtual void Join() = 0;
  virtual bool Probe() = 0;

 protected:
  std::string name_;
  size_t size_;
  size_t nbytes_;
};

class AbstractSendPort : public AbstractPort {
 public:
  using AbstractPort::AbstractPort;
  virtual ~AbstractSendPort() = default;
  virtual void Send(MetaDataPtr data) = 0;
};

class AbstractRecvPort : public AbstractPort {
 public:
  using AbstractPort::AbstractPort;
  virtual ~AbstractRecvPort() = default;
  virtual MetaDataPtr Recv() = 0;
  virtual MetaDataPtr Peek() = 0;
};

using AbstractSendPortPtr = std::shared_ptr<AbstractSendPort>;
using AbstractRecvPortPtr = std::shared_ptr<AbstractRecvPort>;

}  // namespace message_infrastructure

#endif  // ABSTRACT_PORT_H_
