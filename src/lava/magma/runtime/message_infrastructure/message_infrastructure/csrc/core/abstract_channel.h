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
    return NULL;
  }
  virtual AbstractRecvPortPtr GetRecvPort() {
    return NULL;
  }
};

using AbstractChannelPtr = std::shared_ptr<AbstractChannel>;

}  // namespace message_infrastructure

#endif  // CORE_ABSTRACT_CHANNEL_H_
