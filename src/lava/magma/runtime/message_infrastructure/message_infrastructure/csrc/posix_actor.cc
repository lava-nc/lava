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
  this->status_ = StatsStopped;

  // Check the status
  return 0;

}

int PosixActor::Stop() {
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
  this->status_ = StatsStopped;
  return 0;
}

int PosixActor::ReStart() {
  if (this->status_ == StatsStopped) {
    LAVA_LOG(LOG_MP, "Actor Restart, pid: %d\n", this->pid_);
    pid_t pid = fork();
    if (pid == 0) {
      target_fn_();
      exit(0);
    }
    if (pid > 0) {
      this->status_ = StatsRuning;
      this->pid_ = pid;
      LAVA_LOG(LOG_MP, "Actor Restart, new pid: %d\n", this->pid_);
      return ParentProcess;
    }
  }
  else {
    LAVA_LOG_ERR("The Actor is running or paused, no need to restart\n");
    return -1;
  }

  LAVA_LOG_ERR("Actor Restart false\n");
  return ErrorProcess;

}

}