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
#include "port_proxy.h"

namespace message_infrastructure {
class ChannelProxy {
 public:
  ChannelProxy() {}
  ChannelProxy(const ChannelType &channel_type,
               const size_t &size,
               const size_t &nbytes,
               const std::string &name = "test_channel") {
      ChannelFactory &channel_factory = GetChannelFactory();
      channel_ = channel_factory.GetChannel(channel_type,
                                            size,
                                            nbytes,
                                            name);
      send_port_ = std::make_shared<SendPortProxy>(channel_type,
                                                   channel_->GetSendPort());
      recv_port_ = std::make_shared<RecvPortProxy>(channel_type,
                                                   channel_->GetRecvPort());
  }
  SendPortProxyPtr GetSendPort() {
     return send_port_;
  }
  RecvPortProxyPtr GetRecvPort() {
     return recv_port_;
  }

 private:
  AbstractChannelPtr channel_;
  SendPortProxyPtr send_port_ = NULL;
  RecvPortProxyPtr recv_port_ = NULL;
};
}  // namespace message_infrastructure

#endif  // CHANNEL_PROXY_H_
