// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/socket/socket.h>
#include <sys/socket.h>

namespace message_infrastructure {

SktManager::~SktManager() {
  for (auto it = sockets_.begin(); it != sockets_.end(); it++) {
    close(it->first);
    close(it->second);
  }
  sockets_.clear();
}

SocketPair SktManager::AllocChannelSocket(size_t nbytes) {
  SocketPair skt_pair;
  int socket[2];
  int err = socketpair(AF_LOCAL, SOCK_SEQPACKET, 0, socket);
  if (err == -1) {
      LAVA_LOG_FATAL("Create socket object failed.\n");
  }
  skt_pair.first = socket[0];
  skt_pair.second = socket[1];
  sockets_.push_back(skt_pair);
  return skt_pair;
}

SktManager SktManager::sktm_;

SktManager& GetSktManagerSingleton() {
  SktManager &sktm = SktManager::sktm_;
  return sktm;
}
}  // namespace message_infrastructure
