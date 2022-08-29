// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "multiprocessing.h"
#include "message_infrastructure_logging.h"
#include "utils.h"

#include <sys/wait.h>
#include <unistd.h>

namespace message_infrastructure {

MultiProcessing::MultiProcessing() {
  shmm_ = new SharedMemManager();
}

int MultiProcessing::BuildActor(std::function<int(int)> target_fn) {
  pid_t pid = fork();

  int shmid = shmget(signal_key_, sizeof(char), 0644|IPC_CREAT);
  if (shmid == -1) {
    perror("Shared Memory allocate");
    return ErrorProcess;
  }
  char *signal = (char*) shmat(shmid, NULL, 0);
  if (signal == (void*) -1) {
    perror("Shared Memrory attach");
    return ErrorProcess;
  }
  *signal = StatsRuning;

  if (pid > 0) {
    LAVA_LOG(LOG_MP, "Parent Process, create child process %d\n", pid);
    ActorPtr actor = new PosixActor(pid, target_fn, signal);
    signal_key_++;
    actors_.push_back(actor);
    return ParentProcess;
  }

  if (pid == 0) {
    LAVA_LOG(LOG_MP, "child, new process\n");
    int target_ret;
    do {
      target_ret = target_fn(shmid);
    } while (target_ret);

    exit(0);
  }

  LAVA_LOG_ERR("Cannot allocate new pid for the process\n");
  return ErrorProcess;

}

int MultiProcessing::Stop() {
  int error_cnts = 0;

  for (auto actor : actors_) {
    *(actor->signal_) = StatsStopped;
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
