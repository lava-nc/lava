// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_ABSTRACT_CHANNEL_H_
#define CORE_ABSTRACT_CHANNEL_H_

#include <message_infrastructure/csrc/core/abstract_port.h>
#include <memory>


namespace message_infrastructure {

class AbstractChannel {
 public:
  virtual ~AbstractChannel() = default;
  ChannelType channel_type_;

  virtual AbstractSendPortPtr GetSendPort() {
    return nullptr;
  }
  virtual AbstractRecvPortPtr GetRecvPort() {
    return nullptr;
  }
};

using AbstractChannelPtr = std::shared_ptr<AbstractChannel>;

class GrpcAbstractChannel {
 public:
  virtual ~GrpcAbstractChannel() = default;
  ChannelType channel_type_;

  virtual GrpcAbstractSendPortPtr GetSendPort() {
    return nullptr;
  }
  virtual AbstractRecvPortPtr GetRecvPort() {
    return nullptr;
  }
};

using GrpcAbstractChannelPtr = std::shared_ptr<GrpcAbstractChannel>;

}  // namespace message_infrastructure

#endif  // CORE_ABSTRACT_CHANNEL_H_
