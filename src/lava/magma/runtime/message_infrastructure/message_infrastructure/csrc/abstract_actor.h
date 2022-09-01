// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_ACTOR_H_
#define ABSTRACT_ACTOR_H_

#include <functional>
#include "shm.h"

namespace message_infrastructure {

enum ActorStatus {
  StatusError = -1,
  StatusRunning = 0,
  StatusStopped = 1,
  StatusPaused = 2
};

struct ActorStatusInfo {
  ActorStatus ctl_status;
};

class AbstractActor {
 public:
  virtual int ForceStop() = 0;
  virtual int GetActorStatus() = 0;
  virtual int GetPid() = 0;
  virtual int Run() = 0;  // parent process only
  virtual int Stop() = 0;
  int pid_;
 protected:
  int ReMapActorStatus() {  // child process only
    status_shm_->MemMap();
    return 0;
  }
  ActorStatusInfo *actor_status_;
  SharedMemory *status_shm_;
};

using ActorPtr = AbstractActor *;

class PosixActor : public AbstractActor {
 public:
  explicit PosixActor(std::function<int(ActorPtr)> target_fn, int shmid) {
    this->target_fn_ = target_fn;
    status_shm_ = new SharedMemory(sizeof(ActorStatusInfo), shmid);
    this->actor_status_ =
      reinterpret_cast<ActorStatusInfo*>(status_shm_->MemMap());
  }

  int GetPid() {
    return this->pid_;
  }
  int Wait();
  int ForceStop();
  int GetActorStatus() {
    return this->actor_status_->ctl_status;
  }
  int Stop() {
    this->actor_status_->ctl_status = StatusStopped;
    return 0;
  }
  int Run();
  // int Trace();
 private:
  std::function<int(ActorPtr)> target_fn_ = NULL;
};

using PosixActorPtr = PosixActor *;

}  // namespace message_infrastructure

#endif  // ABSTRACT_ACTOR_H_
