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
  bool CheckURL(const std::string &url) {
    if (url_set.count(url))
      return false;
    url_set.insert(url);
    return true;
  }
  friend GrpcManager &GetGrpcManager();
 private:
  GrpcManager() {}
  static GrpcManager grpcm_;
  std::set<std::string> url_set;
};


GrpcManager& GetGrpcManager();

using GrpcManagerPtr = std::shared_ptr<GrpcManager>;

}  // namespace message_infrastructure

#endif  // CHANNEL_GRPC_GRPC_H_
