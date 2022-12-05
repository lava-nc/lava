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
                             const size_t &nbytes) {
  SocketPair skt = GetSktManagerSingleton().AllocChannelSocket(nbytes);
  send_port_ = std::make_shared<SocketSendPort>(src_name, skt, nbytes);
  recv_port_ = std::make_shared<SocketRecvPort>(dst_name, skt, nbytes);
}

AbstractSendPortPtr SocketChannel::GetSendPort() {
  return send_port_;
}

AbstractRecvPortPtr SocketChannel::GetRecvPort() {
  return recv_port_;
}

std::shared_ptr<SocketChannel> GetSocketChannel(const size_t &nbytes,
                                                const std::string &src_name,
                                                const std::string &dst_name) {
  return (std::make_shared<SocketChannel>(src_name,
                                         dst_name,
                                         nbytes));
}

TempSocketChannel::TempSocketChannel(const std::string &addr_path) {
  addr_path_ = GetSktManagerSingleton().AllocSocketFile(addr_path);
}

std::string TempSocketChannel::ChannelInfo() {
  return addr_path_;
}

AbstractRecvPortPtr TempSocketChannel::GetRecvPort() {
  if (recv_port_ == nullptr) {
    recv_port_ = std::make_shared<TempSocketRecvPort>(addr_path_);
  }
  return recv_port_;
}

AbstractSendPortPtr TempSocketChannel::GetSendPort() {
  if (send_port_ == nullptr) {
    send_port_ = std::make_shared<TempSocketSendPort>(addr_path_);
  }
  return send_port_;
}

bool TempSocketChannel::Close() {
  return GetSktManagerSingleton().DeleteSocketFile(addr_path_);
}

}  // namespace message_infrastructure
