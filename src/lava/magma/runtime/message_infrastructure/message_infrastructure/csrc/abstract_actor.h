// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_ACTOR_H_
#define ABSTRACT_ACTOR_H_

#include <functional>
#include <string>
#include <memory>
#include "shm.h"

namespace message_infrastructure {

enum ActorType {
  RuntimeActor = 0,
  RuntimeServiceActor = 1,
  ProcessModelActor = 2
};

enum ActorStatus {
  StatusError = -1,
  StatusRunning = 0,
  StatusStopped = 1,
  StatusPaused = 2
};

enum ActorCmd {
  CmdRun = 0,
  CmdStop = -1,
  CmdPause = -2
};

struct ActorCtrlStatus {
  ActorCmd cmd;
  ActorStatus status;
};

class AbstractActor {
 public:
  virtual int ForceStop() = 0;
  virtual int GetActorStatus() = 0;
  virtual int GetPid() = 0;
  virtual int Create() = 0;  // parent process only
  virtual int ActorControl(int) = 0;  // parent process only
  virtual int ErrorOccured() = 0;  // child process only
 protected:
  int ReMapActorStatus() {  // child process only
    ctl_status_shm_->MemMap();
    return 0;
  }
  int pid_;
  ActorCtrlStatus *actor_ctrl_status_;
  SharedMemoryPtr ctl_status_shm_;
  ActorType actor_type_ = ActorType::ProcessModelActor;
  std::string actor_name_ = "actor";
};

using ActorPtr = AbstractActor *;
using SharedActorPtr = std::shared_ptr<AbstractActor>;

class PosixActor : public AbstractActor {
 public:
  explicit PosixActor(std::function<int(ActorPtr)> target_fn, int shmid) {
    this->target_fn_ = target_fn;
    ctl_status_shm_ = GetSharedMemManager().AllocChannelSharedMemory(
                        sizeof(ActorCtrlStatus));
    ctl_status_shm_->InitSemaphore();
    this->actor_ctrl_status_ =
      reinterpret_cast<ActorCtrlStatus*>(ctl_status_shm_->MemMap());
  }

  int GetPid() {
    return this->pid_;
  }
  int Wait();
  int ForceStop();
  int GetActorStatus();
  int ActorControl(int);
  int ErrorOccured() {
    this->actor_ctrl_status_->status = ActorStatus::StatusError;
    return 0;
  }
  int Create();
  // int Trace();
 private:
  std::function<int(ActorPtr)> target_fn_ = NULL;
};

using PosixActorPtr = PosixActor *;

}  // namespace message_infrastructure

#endif  // ABSTRACT_ACTOR_H_
