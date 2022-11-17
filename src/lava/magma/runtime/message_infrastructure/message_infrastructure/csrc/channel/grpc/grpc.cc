// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/grpc/grpc.h>
#include <mutex>  // NOLINT

namespace message_infrastructure {

GrpcManager GrpcManager::grpcm_;
GrpcManager::~GrpcManager() {
  url_set_.clear();
}
GrpcManager& GetGrpcManagerSingleton() {
  GrpcManager &grpcm = GrpcManager::grpcm_;
  return grpcm;
}
bool GrpcManager::CheckURL(const std::string &url) {
  std::lock_guard<std::mutex> lg(grpc_lock_);
  if (url_set_.find(url) != url_set_.end()) {
    return false;
  }
  url_set_.insert(url);
  return true;
}
std::string GrpcManager::AllocURL() {
  uint32_t port_num = port_num_.load();
  std::string url = DEFAULT_GRPC_URL +
                    std::to_string(DEFAULT_GRPC_PORT + port_num);
  while (!CheckURL(url)) {
    url = DEFAULT_GRPC_URL + std::to_string(DEFAULT_GRPC_PORT + port_num);
    port_num_.store(port_num + 1);
  }
  return url;
}
void GrpcManager::Release() {
  url_set_.clear();
}

}  // namespace message_infrastructure
