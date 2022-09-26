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
#include "port_proxy.h"

namespace message_infrastructure {
class ChannelProxy {
 public:
  ChannelProxy(const ChannelType &channel_type,
               const size_t &size,
               const size_t &nbytes,
               const std::string &name = "test_channel");
  SendPortProxyPtr GetSendPort();
  RecvPortProxyPtr GetRecvPort();
 private:
  AbstractChannelPtr channel_ = NULL;
  SendPortProxyPtr send_port_ = NULL;
  RecvPortProxyPtr recv_port_ = NULL;
};
}  // namespace message_infrastructure

#endif  // CHANNEL_PROXY_H_
