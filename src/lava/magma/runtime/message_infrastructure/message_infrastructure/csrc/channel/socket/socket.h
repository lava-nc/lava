// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_SOCKET_SOCKET_H_
#define CHANNEL_SOCKET_SOCKET_H_

#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <message_infrastructure/csrc/core/utils.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <memory>
#include <unordered_set>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <utility>

#define SKT_TEMP_PATH "/tmp/skt_tmp_"
#define MAX_SKT_FILENAME_LENGTH 100

namespace message_infrastructure {

using SocketPair = std::pair<int, int>;
using SocketFile = std::string;

class SktManager {
 public:
  SktManager(const SktManager&) = delete;
  SktManager(SktManager&&) = delete;
  SktManager& operator=(const SktManager&) = delete;
  SktManager& operator=(SktManager&&) = delete;

  SocketPair AllocChannelSocket(size_t nbytes);
  SocketFile AllocSocketFile(const std::string &addr_path);
  bool DeleteSocketFile(const std::string &addr_path);

  friend SktManager &GetSktManagerSingleton();

 private:
  SktManager() = default;
  ~SktManager();
  std::vector<SocketPair> sockets_;
  std::unordered_set<SocketFile> socket_files_;
  static SktManager sktm_;
};

SktManager& GetSktManagerSingleton();

}  // namespace message_infrastructure

#endif  // CHANNEL_SOCKET_SOCKET_H_
