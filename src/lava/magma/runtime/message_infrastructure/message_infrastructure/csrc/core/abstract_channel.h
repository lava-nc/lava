// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_CHANNEL_H_
#define ABSTRACT_CHANNEL_H_

#include <memory>

#include <message_infrastructure/csrc/core/abstract_port.h>

namespace message_infrastructure {

class AbstractChannel {
 public:
  virtual ~AbstractChannel() = default;
  ChannelType channel_type_;

  virtual AbstractSendPortPtr GetSendPort() {
    return NULL;
  }
  virtual AbstractRecvPortPtr GetRecvPort() {
    return NULL;
  }
};

using AbstractChannelPtr = std::shared_ptr<AbstractChannel>;

}  // namespace message_infrastructure

#endif  // ABSTRACT_CHANNEL_H_
