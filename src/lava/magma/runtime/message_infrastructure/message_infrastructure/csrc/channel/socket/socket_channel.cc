// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/socket/socket_channel.h>
#include <message_infrastructure/csrc/channel/socket/socket_port.h>
#include <message_infrastructure/csrc/channel/socket/socket.h>
#include <message_infrastructure/csrc/core/utils.h>

namespace message_infrastructure {

SocketChannel::SocketChannel(const std::string &src_name,
                             const std::string &dst_name,
                             const size_t &nbytes,
                             const size_t &size) {
  SocketPair skt = GetSktManager().AllocChannelSocket(nbytes);
  send_port_ = std::make_shared<SocketSendPort>(src_name, skt, nbytes);
  recv_port_ = std::make_shared<SocketRecvPort>(dst_name, skt, nbytes);
}

AbstractSendPortPtr SocketChannel::GetSendPort() {
  return send_port_;
}

AbstractRecvPortPtr SocketChannel::GetRecvPort() {
  return recv_port_;
}

std::shared_ptr<SocketChannel> GetSocketChannel(const size_t &size,
                              const size_t &nbytes,
                              const std::string &src_name,
                              const std::string &dst_name) {
  return (std::make_shared<SocketChannel>(src_name,
                                         dst_name,
                                         nbytes,
                                         size));
}

}  // namespace message_infrastructure
