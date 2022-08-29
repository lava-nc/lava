// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef MULTIPROCESSING_H_
#define MULTIPROCESSING_H_

#include <vector>
#include <functional>

#include "abstract_actor.h"
#include "shm.h"

namespace message_infrastructure {

class MultiProcessing {
 public:
  MultiProcessing();
  int Stop();
  int BuildActor(std::function<int(int)>);
  void CheckActor();
  std::vector<ActorPtr>& GetActors();
  SharedMemManager* GetSharedMemManager();

 private:
  std::vector<ActorPtr> actors_;
  int signal_key_ = 0xbee;
  SharedMemManager *shmm_;
};

}  // namespace message_infrastructure

#endif  // MULTIPROCESSING_H_
