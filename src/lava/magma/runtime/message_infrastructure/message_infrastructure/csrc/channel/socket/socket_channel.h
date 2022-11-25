// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_SOCKET_SOCKET_CHANNEL_H_
#define CHANNEL_SOCKET_SOCKET_CHANNEL_H_

#include <message_infrastructure/csrc/channel/socket/socket_port.h>
#include <message_infrastructure/csrc/channel/socket/socket.h>
#include <message_infrastructure/csrc/core/abstract_channel.h>
#include <message_infrastructure/csrc/core/abstract_port.h>

#include <memory>
#include <string>

namespace message_infrastructure {

class SocketChannel : public AbstractChannel {
 public:
  SocketChannel() = delete;
  SocketChannel(const std::string &src_name,
                const std::string &dst_name,
                const size_t &nbytes);
  ~SocketChannel() override {}
  AbstractSendPortPtr GetSendPort();
  AbstractRecvPortPtr GetRecvPort();
 private:
  SocketPair skt_;
  SocketSendPortPtr send_port_ = nullptr;
  SocketRecvPortPtr recv_port_ = nullptr;
};

// Users should be allowed to copy channel objects.
// Use std::shared_ptr.
using SocketChannelPtr = std::shared_ptr<SocketChannel>;

SocketChannelPtr GetSocketChannel(const size_t &nbytes,
                                  const std::string &src_name,
                                  const std::string &dst_name);

class TempSocketChannel : public AbstractChannel {
 public:
  TempSocketChannel() = delete;
  TempSocketChannel(const std::string &addr_path);
  ~TempSocketChannel() override {}
  AbstractSendPortPtr GetSendPort();
  AbstractRecvPortPtr GetRecvPort();
  std::string ChannelInfo() override;
  bool Close();
 private:
  SocketFile addr_path_;
  TempSocketRecvPortPtr recv_port_ = nullptr;
  TempSocketSendPortPtr send_port_ = nullptr;
};

}  // namespace message_infrastructure

#endif  // CHANNEL_SOCKET_SOCKET_CHANNEL_H_
