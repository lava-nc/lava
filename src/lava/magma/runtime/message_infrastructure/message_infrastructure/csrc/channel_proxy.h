// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_PROXY_H_
#define CHANNEL_PROXY_H_

#include <message_infrastructure/csrc/core/abstract_channel.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/port_proxy.h>

#include <string>

namespace message_infrastructure {

class ChannelProxy {
 public:
  ChannelProxy(const ChannelType &channel_type,
               const size_t &size,
               const size_t &nbytes,
               const std::string &src_name,
               const std::string &dst_name);
  SendPortProxyPtr GetSendPort();
  RecvPortProxyPtr GetRecvPort();
 private:
  AbstractChannelPtr channel_ = nullptr;
  SendPortProxyPtr send_port_ = nullptr;
  RecvPortProxyPtr recv_port_ = nullptr;
};

}  // namespace message_infrastructure

#endif  // CHANNEL_PROXY_H_
