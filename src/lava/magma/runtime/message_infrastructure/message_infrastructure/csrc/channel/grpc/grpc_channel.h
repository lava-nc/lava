// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_GRPC_GRPC_CHANNEL_H_
#define CHANNEL_GRPC_GRPC_CHANNEL_H_

#include <message_infrastructure/csrc/channel/grpc/grpc_port.h>
#include <message_infrastructure/csrc/core/abstract_channel.h>

#include <memory>
#include <string>

namespace message_infrastructure {

class GrpcChannel : public AbstractChannel {
 public:
  GrpcChannel() {}
  GrpcChannel(const std::string &src_name,
               const std::string &dst_name,
               const size_t &size,
               const size_t &nbytes);
  AbstractSendPortPtr GetSendPort();
  AbstractRecvPortPtr GetRecvPort();
 private:
  GrpcSendPortPtr send_port_ = NULL;
  GrpcRecvPortPtr recv_port_ = NULL;
};

using GrpcChannelPtr = std::shared_ptr<GrpcChannel>;

GrpcChannelPtr GetGrpcChannel(const size_t &size,
                              const size_t &nbytes,
                              const std::string &src_name,
                              const std::string &dst_name);

}  // namespace message_infrastructure

#endif  // CHANNEL_GRPC_GRPC_CHANNEL_H_
