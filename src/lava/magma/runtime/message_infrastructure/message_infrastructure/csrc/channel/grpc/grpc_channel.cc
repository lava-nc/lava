// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/



#include <message_infrastructure/csrc/channel/grpc/grpc_channel.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/channel/grpc/grpc.h>

#include <string>
#include <memory>

namespace message_infrastructure {

GrpcChannel::GrpcChannel(const std::string &src_name,
                           const std::string &dst_name,
                           const size_t &size,
                           const size_t &nbytes) {
  std::string url = GetGrpcManager().AllocChannelGrpc(nbytes);
  send_port_ = std::make_shared<GrpcSendPort>(src_name, size, nbytes, url);
  recv_port_ = std::make_shared<GrpcRecvPort>(dst_name, size, nbytes, url);
}
AbstractSendPortPtr GrpcChannel::GetSendPort() {
  return send_port_;
}

AbstractRecvPortPtr GrpcChannel::GetRecvPort() {
  return recv_port_;
}

std::shared_ptr<GrpcChannel> GetGrpcChannel(const size_t &size,
                              const size_t &nbytes,
                              const std::string &src_name,
                              const std::string &dst_name) {
  return (std::make_shared<GrpcChannel>(src_name,
                                         dst_name,
                                         size,
                                         nbytes));
}

}   // namespace message_infrastructure
