// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/grpc/grpc.h>

namespace message_infrastructure {

GrpcManager GrpcManager::grpcm_;
GrpcManager::~GrpcManager() {
  port_num_ = 0;
  url_num_ = 0;
  urls_.clear();
}
GrpcManager& GetGrpcManager() {
  GrpcManager &grpcm = GrpcManager::grpcm_;
  return grpcm;
}

}  // namespace message_infrastructure
