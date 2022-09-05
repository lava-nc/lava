// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_PROXY_H_
#define CHANNEL_PROXY_H_

#include <memory>
#include <string>
#include <vector>
#include "abstract_channel.h"
#include "utils.h"
#include "channel_factory.h"

namespace message_infrastructure {
class ChannelProxy {
 public:
  ChannelProxy() {}
  ChannelProxy(const ChannelType &channel_type,
      const SharedMemManager &smm,
      const size_t &size,
      const size_t &nbytes,
      const std::string &name = "test_channel") {
    ChannelFactory &channel_factory = GetChannelFactory();
    channel_ = channel_factory.GetChannel(channel_type,
                                          smm,
                                          name,
                                          size,
                                          nbytes);
  }
  SendPortProxyPtr GetSendPort() {
     return channel_->GetSendPort();
  }
  RecvPortProxyPtr GetRecvPort() {
     return channel_->GetRecvPort();
  }
 private:
  AbstractChannelPtr channel_;
}
}  // namespace message_infrastructure

#endif  // CHANNEL_PROXY_H_
