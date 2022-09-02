// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_actor.h"
#include "message_infrastructure_logging.h"
#include "utils.h"

#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>

namespace message_infrastructure {

int PosixActor::Wait() {
  int status;
  // TODO: Add the options can be as args of the function
  int options = 0;
  int ret = waitpid(this->pid_, &status, options);

  if (ret < 0) {
    LAVA_LOG_ERR("process %d waitpid error\n", this->pid_);
    return -1;
  }
  this->actor_status_->ctl_status = StatusStopped;

  // Check the status
  return 0;

}

int PosixActor::ForceStop() {
  int status;
  kill(pid_, SIGTERM);
  wait(&status);
  if (WIFSIGNALED(status)) {
    if (WTERMSIG(status) == SIGTERM) {
      LAVA_LOG(LOG_MP, "The Actor child was ended with SIGTERM\n");
    }
    else {
      LAVA_LOG(LOG_MP, "The Actor child was ended with signal %d\n", status);
    }
  }
  this->actor_status_->ctl_status = StatusStopped;
  return 0;
}

int PosixActor::Run() {
  pid_t pid = fork();
  if (pid > 0) {
    LAVA_LOG(LOG_MP, "Parent Process, create child process %d\n", pid);
    this->actor_status_->ctl_status = StatusRunning;
    this->pid_ = pid;
    return ParentProcess;
  }

  if (pid == 0) {
    LAVA_LOG(LOG_MP, "child, new process %d\n", getpid());
    this->pid_ = getpid();
    ReMapActorStatus();
    int target_ret;
    do {
      target_ret = target_fn_(this);
    } while (target_ret && (this->actor_status_->ctl_status != StatusStopped));
    LAVA_LOG(LOG_MP, "child exist\n");
    exit(0);
  }
  LAVA_LOG_ERR("Cannot allocate new pid for the process\n");
  return ErrorProcess;
}

}