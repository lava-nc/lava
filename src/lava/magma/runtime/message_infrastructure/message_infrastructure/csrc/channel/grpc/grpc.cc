// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/
#include <message_infrastructure/csrc/channel/grpc/grpc.h>

namespace message_infrastructure {

GrpcManager::~GrpcManager() {
  port_num = 0;
  url_num = 0;
  urls_.clear();
}

GrpcManager GrpcManager::grpcm_;

GrpcManager& GetGrpcManager() {
  GrpcManager &grpcm = GrpcManager::grpcm_;
  return grpcm;
}
}  // namespace message_infrastructure
