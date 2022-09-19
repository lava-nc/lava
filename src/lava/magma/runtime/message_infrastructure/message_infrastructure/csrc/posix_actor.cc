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

int CheckSemaphore(sem_t *sem) {
  int sem_val;
  sem_getvalue(sem, &sem_val);
  if(sem_val < 0) {
    LAVA_LOG_ERR("get the negtive sem value: %d\n", sem_val);
    return -1;
  }
  if(sem_val == 1) {
    LAVA_LOG_ERR("There is a semaphere not used\n");
    return 1;
  }
  
  return 0;
}

int PosixActor::Wait() {
  int status;
  // TODO: Add the options can be as args of the function
  int options = 0;
  int ret = waitpid(this->pid_, &status, options);

  if (ret < 0) {
    LAVA_LOG_ERR("process %d waitpid error\n", this->pid_);
    return -1;
  }

  LAVA_DEBUG(LOG_ACTOR, "current actor status: %d\n", GetStatus());
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
  SetStatus(ActorStatus::StatusStopped);
  return 0;
}

int PosixActor::Create() {
  pid_t pid = fork();
  if (pid > 0) {
    LAVA_LOG(LOG_MP, "Parent Process, create child process %d\n", pid);
    this->pid_ = pid;
    return ParentProcess;
  }

  if (pid == 0) {
    LAVA_LOG(LOG_MP, "child, new process %d\n", getpid());
    this->pid_ = getpid();
    SetStatus(ActorStatus::StatusRunning);
    Control(ActorCmd::CmdRun);
    while(!HandleCmd()) {
      int target_ret = target_fn_(this);
      // if (target_ret >= 0) {
      //   break;
      //}
      LAVA_LOG(LOG_MP, "Actor: target_ret:%d, ActorStatus:%d\n",
                     target_ret, GetStatus());
    }
    SetStatus(ActorStatus::StatusStopped);
    LAVA_LOG(LOG_ACTOR, "child exist, pid:%d\n", this->pid_);
    exit(0);
  }
  LAVA_LOG_ERR("Cannot allocate new pid for the process\n");
  return ErrorProcess;
}

int PosixActor::CmdRun() {
  Control(ActorCmd::CmdRun);
  return 0;
}

int PosixActor::CmdPause() {
  Control(ActorCmd::CmdPause);
  return 0;
}

int PosixActor::CmdStop() {
  Control(ActorCmd::CmdStop);
  return 0;
}

int PosixActor::GetActorStatus() {
  return GetStatus();
}

}