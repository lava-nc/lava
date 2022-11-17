// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_GRPC_GRPC_H_
#define CHANNEL_GRPC_GRPC_H_

#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

#include <string>
#include <unordered_set>
#include <memory>

namespace message_infrastructure {

#define DEFAULT_GRPC_URL "0.0.0.0:"
#define DEFAULT_GRPC_PORT 8000

class GrpcManager {
 public:
  GrpcManager(const GrpcManager&) = delete;
  GrpcManager(GrpcManager&&) = delete;
  GrpcManager& operator=(const GrpcManager&) = delete;
  GrpcManager& operator=(GrpcManager&&) = delete;

  bool CheckURL(const std::string &url) {
    if (url_set_.find(url) != url_set_.end()) {
      return false;
    }
    url_set_.insert(url);
    return true;
  }
  std::string AllocURL() {
    std::string url = DEFAULT_GRPC_URL +
      std::to_string(DEFAULT_GRPC_PORT + port_num_);
    while (!CheckURL(url)) {
      url = DEFAULT_GRPC_URL + std::to_string(DEFAULT_GRPC_PORT + port_num_);
      port_num_++;
    }
    return url;
  }
  void Release() {
    url_set_.clear();
  }
  friend GrpcManager &GetGrpcManagerSingleton();

 private:
  GrpcManager() = default;
  ~GrpcManager();
  int port_num_ = 0;
  static GrpcManager grpcm_;
  std::unordered_set<std::string> url_set_;
};

GrpcManager& GetGrpcManagerSingleton();

}  // namespace message_infrastructure

#endif  // CHANNEL_GRPC_GRPC_H_
