// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "multiprocessing.h"
#include "message_infrastructure_logging.h"
#include "utils.h"


namespace message_infrastructure {

int MultiProcessing::BuildActor(AbstractActor::TargetFn target_fn) {
  SharedMemManager &actor_shmm = GetSharedMemManager();
  int shmid = actor_shmm.AllocSharedMemory(sizeof(ActorCtrlStatus));
  AbstractActor::ActorPtr actor = new PosixActor(target_fn, shmid);
  int ret = actor->Create();
  actors_.push_back(actor);
  return ret;
}

int MultiProcessing::Stop() {
  int error_cnts = 0;

  for (auto actor : actors_) {
    actor->Control(ActorCmd::CmdStop);
  }

  LAVA_LOG(LOG_MP, "Stop Actors, error: %d\n", error_cnts);
  return error_cnts;
}

void MultiProcessing::CheckActor() {
  for (auto actor : actors_) {
    LAVA_LOG(LOG_MP, "Actor info: (pid, status):(%d, %d)", 
                     actor->GetPid(), actor->GetStatus());
  }
}

std::vector<AbstractActor::ActorPtr>& MultiProcessing::GetActors() {
  return this->actors_;
}

}  // namespace message_infrastructure
