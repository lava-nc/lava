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

  LAVA_DEBUG(LOG_ACTOR, "current actor status: %d\n", this->actor_ctrl_status_->status);
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
  this->actor_ctrl_status_->status = StatusStopped;
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
    ReMapActorStatus();
    this->actor_ctrl_status_->status = ActorStatus::StatusStopped;
    int target_ret;
    while(true) {
      target_ret = target_fn_(this);
      if (this->actor_ctrl_status_->status == ActorStatus::StatusStopped)
        break;
      else {
        LAVA_LOG_ERR("Actor ERROR: target_ret:%d, ActorStatus:%d\n",
                     target_ret, this->actor_ctrl_status_->status);
      }
    }
    LAVA_LOG(LOG_ACTOR, "child exist, pid:%d\n", this->pid_);
    exit(0);
  }
  LAVA_LOG_ERR("Cannot allocate new pid for the process\n");
  return ErrorProcess;
}

int PosixActor::CmdRun() {
  sem_t *sem = this->ctl_status_shm_->GetAckSemaphore();
  if (CheckSemaphore(sem)) {
    LAVA_LOG_ERR("CmdRun Semaphere check error\n");
    return -1;
  }
  this->actor_ctrl_status_->cmd == ActorCmd::CmdRun;
  sem_post(sem);
  
  sem = this->ctl_status_shm_->GetReqSemaphore();
  sem_wait(sem);
  if (this->actor_ctrl_status_->status == ActorStatus::StatusRunning)
    return 0;
  else {
    LAVA_LOG_ERR("CmdRun Setting Error\n");
  }
}

int PosixActor::CmdPause() {
  sem_t *sem = this->ctl_status_shm_->GetAckSemaphore();
  if (CheckSemaphore(sem)) {
    LAVA_LOG_ERR("CmdPause Semaphere check error\n");
    return -1;
  }
  this->actor_ctrl_status_->cmd == ActorCmd::CmdPause;
  
  sem = this->ctl_status_shm_->GetReqSemaphore();
  sem_wait(sem);
  if (this->actor_ctrl_status_->status == ActorStatus::StatusPaused)
    return 0;
  else {
    LAVA_LOG_ERR("CmdPause Setting Error\n");
  }
}

int PosixActor::CmdStop() {
  sem_t *sem = this->ctl_status_shm_->GetAckSemaphore();
  if (CheckSemaphore(sem)) {
    LAVA_LOG_ERR("CmdStop Semaphere check error\n");
    return -1;
  }
  this->actor_ctrl_status_->cmd == ActorCmd::CmdStop;
  sem_post(sem);

  sem = this->ctl_status_shm_->GetReqSemaphore();
  sem_wait(sem);
  if (this->actor_ctrl_status_->status == ActorStatus::StatusStopped)
    return 0;
  else {
    LAVA_LOG_ERR("CmdStop Setting Error\n");
  }
}

int PosixActor::GetActorStatus() {
  sem_t *sem = this->ctl_status_shm_->GetAckSemaphore();
  int sem_val;
  LAVA_DEBUG(LOG_ACTOR, "check semaphore\n");
  sem_getvalue(sem, &sem_val);
  LAVA_DEBUG(LOG_ACTOR, "checked semaphore, sem_val:%d\n", sem_val);

  if (sem_val == 0) {
    return this->actor_ctrl_status_->status;
  }
  else if (sem_val < 0) {
    LAVA_LOG_ERR("Actor semaphore wrong, %d\n", sem_val);
    this->actor_ctrl_status_->status = ActorStatus::StatusError;
  }
  else {
    sem_wait(sem);
    switch(this->actor_ctrl_status_->cmd) {
      case ActorCmd::CmdPause:
        this->actor_ctrl_status_->status = ActorStatus::StatusPaused;
        break;
      case ActorCmd::CmdRun:
        this->actor_ctrl_status_->status = ActorStatus::StatusRunning;
        break;
      case ActorCmd::CmdStop:
        this->actor_ctrl_status_->status = ActorStatus::StatusStopped;
        break;
      default:
        LAVA_LOG_ERR("Not support the ActorCmd\n");
        break;
    }
  }
  sem = this->ctl_status_shm_->GetReqSemaphore();
  CheckSemaphore(sem);
  sem_post(sem);

  return this->actor_ctrl_status_->status;
}

}