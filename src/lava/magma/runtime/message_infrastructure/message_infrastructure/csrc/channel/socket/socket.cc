// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/socket/socket.h>

namespace message_infrastructure {

SktManager::~SktManager() {
  for (auto it = sockets_.begin(); it != sockets_.end(); it++) {
    close(it->first);
    close(it->second);
  }
  sockets_.clear();
}

SktManager SktManager::sktm_;

SktManager& GetSktManagerSingleton() {
  SktManager &sktm = SktManager::sktm_;
  return sktm;
}
}  // namespace message_infrastructure
