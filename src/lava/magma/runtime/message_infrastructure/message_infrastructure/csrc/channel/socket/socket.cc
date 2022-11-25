// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/socket/socket.h>
#include <sys/socket.h>
#include <filesystem>

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

SocketFile SktManager::AllocSocketFile(const std::string addr_path) {
  SocketFile skt_file;
  if (std::string() == addr_path) {
    LAVA_DEBUG(LOG_SKP, "Creating Socket File\n");
    std::srand(std::time(nullptr));
    do {
      skt_file = SKT_TEMP_PATH + std::to_string(rand());
    } while (std::filesystem::exists(skt_file));
  } else {
    skt_file = addr_path;
  }
  if (socket_files_.find(skt_file) != socket_files_.end()) {
    LAVA_LOG_ERR("Skt File %d is alread used by the process\n", skt_file);
  }
  socket_files_.insert(skt_file);
  return skt_file;
}

bool SktManager::DeleteSocketFile(const std::string addr_path) {
  if (socket_files_.find(addr_path) == socket_files_.end()) {
    LAVA_LOG_WARN(LOG_SKP, "Cannot delete exist file name\n");
    return false;
  }
  socket_files_.erase(addr_path);
  return true;
}

SktManager SktManager::sktm_;

SktManager& GetSktManagerSingleton() {
  SktManager &sktm = SktManager::sktm_;
  return sktm;
}
}  // namespace message_infrastructure
