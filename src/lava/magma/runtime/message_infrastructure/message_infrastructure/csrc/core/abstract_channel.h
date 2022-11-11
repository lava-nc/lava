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

// Users should be allowed to copy channel objects.
// Use std::shared_ptr.
using AbstractChannelPtr = std::shared_ptr<AbstractChannel>;

#if defined(GRPC_CHANNEL)
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
#endif
}  // namespace message_infrastructure

#endif  // CORE_ABSTRACT_CHANNEL_H_
