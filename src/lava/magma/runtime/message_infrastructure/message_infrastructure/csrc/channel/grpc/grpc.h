// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_GRPC_GRPC_H_
#define CHANNEL_GRPC_GRPC_H_

#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <memory>
#include <set>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>

namespace message_infrastructure {

class GrpcManager {
 public:
  ~GrpcManager();

  std::string AllocChannelGrpc(size_t nbytes) {
    std::string url = base_url + std::to_string(url_num) \
                      + base_port+std::to_string(port_num);
    urls_.push_back(url);
    url_num++;
    port_num++;
    return url;
  }

  friend GrpcManager &GetGrpcManager();

 private:
  GrpcManager() {}
  std::string base_url = "127.13.2.";
  std::string base_port = ":800";
  int url_num = 78;
  int port_num = 0;
  std::vector<std::string> urls_;
  static GrpcManager grpcm_;
};


GrpcManager& GetGrpcManager();

using GrpcManagerPtr = std::shared_ptr<GrpcManager>;

}  // namespace message_infrastructure

#endif  // CHANNEL_GRPC_GRPC_H_
