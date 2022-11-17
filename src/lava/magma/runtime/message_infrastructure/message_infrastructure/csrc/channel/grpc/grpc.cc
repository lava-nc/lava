// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/grpc/grpc.h>

namespace message_infrastructure {

GrpcManager GrpcManager::grpcm_;
GrpcManager::~GrpcManager() {
  url_set_.clear();
}
GrpcManager& GetGrpcManagerSingleton() {
  GrpcManager &grpcm = GrpcManager::grpcm_;
  return grpcm;
}

}  // namespace message_infrastructure
