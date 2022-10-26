// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_GRPC_GRPC_H_
#define CHANNEL_GRPC_GRPC_H_

#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

#include <string>
#include <vector>
#include <set>
#include <memory>

namespace message_infrastructure {

class GrpcManager {
 public:
  ~GrpcManager();
  bool CheckURL(const std::string &url) {
    if (url_set_.count(url))
      return false;
    url_set_.insert(url);
    return true;
  }

  std::string AllocURL() {
    std::string url = base_url_ + std::to_string(url_num_) \
                      + base_port_+std::to_string(port_num_);
    urls_.push_back(url);
    url_num_++;
    port_num_++;
    return url;
  }
  friend GrpcManager &GetGrpcManager();

 private:
  GrpcManager() {}
  std::string base_url_ = "127.13.2.";
  std::string base_port_ = ":800";
  int url_num_ = 78;
  int port_num_ = 0;
  std::vector<std::string> urls_;
  static GrpcManager grpcm_;
  std::set<std::string> url_set_;
};


GrpcManager& GetGrpcManager();

using GrpcManagerPtr = std::shared_ptr<GrpcManager>;

}  // namespace message_infrastructure

#endif  // CHANNEL_GRPC_GRPC_H_
