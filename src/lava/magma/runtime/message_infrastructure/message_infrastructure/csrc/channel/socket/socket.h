// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SOCKET_H_
#define SOCKET_H_

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <memory>
#include <set>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>

#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <message_infrastructure/csrc/core/utils.h>

namespace message_infrastructure {

using SocketPair = std::pair<int, int>;

class SktManager
{
public:
  ~SktManager();

  SocketPair AllocChannelSocket(size_t nbytes) {
    SocketPair skt_pair;
    int socket[2];
    int err = socketpair(AF_LOCAL, SOCK_SEQPACKET, 0, socket);
    if (err == -1){
        LAVA_LOG_ERR("Create socket object failed.\n");
        exit(-1);
    }
    skt_pair.first = socket[0];
    skt_pair.second = socket[1];
    sockets_.push_back(skt_pair);
    return skt_pair;
  }

  friend SktManager &GetSktManager();

 private:
  SktManager() {}
  std::vector<SocketPair> sockets_;
  static SktManager sktm_;
};

SktManager& GetSktManager();

using SktManagerPtr = std::shared_ptr<SktManager>;

}  // namespace message_infrastructure

#endif  // SOCKET_H_
