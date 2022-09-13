// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_PORT_H_
#define ABSTRACT_PORT_H_

#include <pybind11/pybind11.h>
#include <string>
#include <vector>
#include <memory>

#include "shm.h"
#include "utils.h"

namespace message_infrastructure {
class AbstractPort {
 public:
  AbstractPort() = default;
  virtual ~AbstractPort() = default;

  virtual std::string Name();
  virtual size_t Size();
  virtual void Start();
  virtual void Join();
  virtual bool Probe();

  std::string name_;
  size_t size_;
  size_t nbytes_;
};

class AbstractSendPort : public AbstractPort {
 public:
  virtual ~AbstractSendPort() = default;
  virtual std::string Name();
  virtual size_t Size();
  virtual void Start();
  virtual void Join();
  virtual bool Probe();
  virtual void Send(MetaDataPtr data);
};

class AbstractRecvPort : public AbstractPort {
 public:
  virtual ~AbstractRecvPort() = default;
  virtual std::string Name();
  virtual size_t Size();
  virtual void Start();
  virtual bool Probe();
  virtual void Join();
  virtual MetaDataPtr Recv();
  virtual MetaDataPtr Peek();
};

using AbstractSendPortPtr = std::shared_ptr<AbstractSendPort>;
using AbstractRecvPortPtr = std::shared_ptr<AbstractRecvPort>;

}  // namespace message_infrastructure

#endif  // ABSTRACT_PORT_H_
