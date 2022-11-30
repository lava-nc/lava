// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/core/multiprocessing.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <message_infrastructure/csrc/actor/posix_actor.h>
#if defined(GRPC_CHANNEL)
#include <message_infrastructure/csrc/channel/grpc/grpc.h>
#endif

namespace message_infrastructure {

ProcessType MultiProcessing::BuildActor(AbstractActor::TargetFn target_fn) {
  AbstractActor::ActorPtr actor = new PosixActor(target_fn);
  ProcessType ret = actor->Create();
  actors_.push_back(actor);
  return ret;
}

void MultiProcessing::Stop() {
  for (auto actor : actors_) {
    actor->Control(ActorCmd::CmdStop);
  }
  LAVA_LOG(LOG_MP, "Send Stop cmd to Actors\n");
}

void MultiProcessing::Cleanup(bool block) {
  if (block) {
    for (auto actor : actors_) {
      actor->Wait();
    }
  }
  GetSharedMemManagerSingleton().DeleteAllSharedMemory();
#if defined(GRPC_CHANNEL)
  GetGrpcManagerSingleton().Release();
#endif
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
  GetSharedMemManagerSingleton().DeleteAllSharedMemory();
}

}  // namespace message_infrastructure
