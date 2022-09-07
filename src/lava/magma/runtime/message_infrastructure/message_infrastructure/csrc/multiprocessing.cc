// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "multiprocessing.h"
#include "message_infrastructure_logging.h"
#include "utils.h"


namespace message_infrastructure {

MultiProcessing::MultiProcessing() {
  int key = 0xbeef;
  int offset = 0x1000;
  shmm_ = new SharedMemManager(key);
  actor_shmm_ = new SharedMemManager(key+offset);
}

int MultiProcessing::BuildActor(std::function<int(ActorPtr)> target_fn) {
  int shmid = actor_shmm_->AllocSharedMemory(sizeof(ActorStatusInfo));
  ActorPtr actor = new PosixActor(target_fn, shmid);
  int ret = actor->Run();
  actors_.push_back(actor);
  
  return ret;
}

int MultiProcessing::Stop() {
  int error_cnts = 0;

  for (auto actor : actors_) {
    error_cnts += actor->Stop();
  }

  LAVA_LOG(LOG_MP, "Stop Actors, error: %d\n", error_cnts);
  return error_cnts;
}

void MultiProcessing::CheckActor() {
  for (auto actor : actors_) {
    LAVA_LOG(LOG_MP, "Actor info %d\n", actor->GetPid());
  }
}

SharedMemManager* MultiProcessing::GetSharedMemManager() {
  return this->shmm_;
}

std::vector<ActorPtr>& MultiProcessing::GetActors() {
  return this->actors_;
}

} // namespace message_infrastructure
