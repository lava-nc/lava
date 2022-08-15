// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "multiprocessing.h"
#include "message_infrastructure_logging.h"

#include <sys/wait.h>
#include <unistd.h>

namespace message_infrastructure {

int MultiProcessing::BuildActor() {
  pid_t pid = fork();

  if (pid > 0) {
    LAVA_LOG(LOG_MP, "Parent Process, create child process %d\n", pid);
    ActorPtr actor = new PosixActor(pid);
    actors_.push_back(actor);
    return PARENT_PROCESS;
  }

  if (pid == 0) {
    LAVA_LOG(LOG_MP, "child, new process\n");
    return CHILD_PROCESS;
  }

  LAVA_LOG_ERR("Cannot allocate new pid for the process\n");
  return ERROR_PROCESS;

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

} // namespace message_infrastructure
