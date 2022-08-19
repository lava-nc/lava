// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_CHANNEL_H_
#define ABSTRACT_CHANNEL_H_

#include <memory>

#include "utils.h"
#include "port_proxy.h"

namespace message_infrastructure {

class AbstractChannel {
 public:
  virtual ~AbstractChannel() {}
  ChannelType channel_type_;

  SendPortProxyPtr GetSendPort() {
     return NULL;
  }
  RecvPortProxyPtr GetRecvPort() {
    return NULL;
  }
};

using AbstractChannelPtr = std::shared_ptr<AbstractChannel>;

}  // namespace message_infrastructure

#endif  // ABSTRACT_CHANNEL_H_
