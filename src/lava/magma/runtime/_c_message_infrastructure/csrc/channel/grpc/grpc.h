// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_GRPC_GRPC_H_
#define CHANNEL_GRPC_GRPC_H_

#include <core/utils.h>
#include <core/message_infrastructure_logging.h>

#include <string>
#include <unordered_set>
#include <memory>
#include <atomic>

namespace message_infrastructure {

class GrpcManager {
 public:
  GrpcManager(const GrpcManager&) = delete;
  GrpcManager(GrpcManager&&) = delete;
  GrpcManager& operator=(const GrpcManager&) = delete;
  GrpcManager& operator=(GrpcManager&&) = delete;

  bool CheckURL(const std::string &url);
  std::string AllocURL();
  void Release();
  friend GrpcManager &GetGrpcManagerSingleton();

 private:
  GrpcManager() = default;
  ~GrpcManager();
  std::mutex grpc_lock_;
  std::atomic<uint32_t> port_num_;
  static GrpcManager grpcm_;
  std::unordered_set<std::string> url_set_;
};

GrpcManager& GetGrpcManagerSingleton();

}  // namespace message_infrastructure

#endif  // CHANNEL_GRPC_GRPC_H_
