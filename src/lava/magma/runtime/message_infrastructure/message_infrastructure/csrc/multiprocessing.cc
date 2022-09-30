// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "multiprocessing.h"
#include "message_infrastructure_logging.h"

namespace message_infrastructure {

int MultiProcessing::BuildActor(AbstractActor::TargetFn target_fn) {
  AbstractActor::ActorPtr actor = new PosixActor(target_fn);
  int ret = actor->Create();
  actors_.push_back(actor);
  return ret;
}

void MultiProcessing::Stop(bool block) {
  for (auto actor : actors_) {
    actor->Control(ActorCmd::CmdStop);
  }

  LAVA_LOG(LOG_MP, "Send Stop cmd to Actors\n");
  if (block) {
    for (auto actor: actors_) {
      actor->Wait();
    }
  }
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

MultiProcessing::~MultiProcessing() {
  for (auto actor : actors_) {
    delete actor;
  }
}

}  // namespace message_infrastructure
