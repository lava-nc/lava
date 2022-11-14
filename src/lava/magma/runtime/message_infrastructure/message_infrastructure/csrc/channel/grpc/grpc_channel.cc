// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/grpc/grpc_channel.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/channel/grpc/grpc.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

#include <stdexcept>

namespace message_infrastructure {

GrpcChannel::GrpcChannel(const std::string &url,
                         const int &port,
                         const std::string &src_name,
                         const std::string &dst_name,
                         const size_t &size) {
  std::string url_ = url + ":" + std::to_string(port);
  bool ret = GetGrpcManager().CheckURL(url_);
  if (!ret) {
    LAVA_LOG_ERR("URL is used, Throw an exception\n");
    throw std::invalid_argument(url_ + " is used now!");
  }
  send_port_ = std::make_shared<GrpcSendPort>(src_name, size, url_);
  recv_port_ = std::make_shared<GrpcRecvPort>(dst_name, size, url_);
}

GrpcChannel::GrpcChannel(const std::string &src_name,
                         const std::string &dst_name,
                         const size_t &size) {
  std::string url_ = GetGrpcManager().AllocURL();
  send_port_ = std::make_shared<GrpcSendPort>(src_name, size, url_);
  recv_port_ = std::make_shared<GrpcRecvPort>(dst_name, size, url_);
}

AbstractSendPortPtr GrpcChannel::GetSendPort() {
  return send_port_;
}

AbstractRecvPortPtr GrpcChannel::GetRecvPort() {
  return recv_port_;
}

}   // namespace message_infrastructure
