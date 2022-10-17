// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SOCKET_CHANNEL_H_
#define SOCKET_CHANNEL_H_

#include <memory>
#include <string>

#include "abstract_channel.h"
#include "abstract_port.h"
#include "socket.h"
#include "socket_port.h"

namespace message_infrastructure {

class SocketChannel : public AbstractChannel {
 public:
  SocketChannel() {}
  SocketChannel(const std::string &src_name,
                const std::string &dst_name,
                const size_t &size,
                const size_t &nbytes);
  AbstractSendPortPtr GetSendPort();
  AbstractRecvPortPtr GetRecvPort();
 private:
  SocketPair skt_;
  SocketSendPortPtr send_port_ = NULL;
  SocketRecvPortPtr recv_port_ = NULL;
};

using SocketChannelPtr = std::shared_ptr<SocketChannel>;

SocketChannelPtr GetSocketChannel(const size_t &size,
                                  const size_t &nbytes,
                                  const std::string &src_name,
                                  const std::string &dst_name);

}  // namespace message_infrastructure

#endif  // SHMEM_CHANNEL_H_
