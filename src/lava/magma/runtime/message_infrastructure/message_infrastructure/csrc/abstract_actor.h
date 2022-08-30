// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_ACTOR_H_
#define ABSTRACT_ACTOR_H_

#include <functional>

namespace message_infrastructure {

enum ActorStatus {
  StatusError = -1,
  StatusRuning = 0,
  StatusStopped = 1
};

class AbstractActor {
 public:
  virtual int GetPid() = 0;
  virtual int Stop() = 0;
  int pid_;
};

class PosixActor : public AbstractActor {
 public:
  explicit PosixActor(int pid, std::function<void()> target_fn) {
    this->pid_ = pid;
    this->target_fn_ = target_fn;
  }
  int GetPid() {
    return this->pid_;
  }
  int Wait();
  int Stop();
  int GetStatus() {
    return this->status_;
  }
  // int Trace();
 private:
  std::function<void()> target_fn_ = NULL;
  int status_ = StatusStopped;
};

using ActorPtr = AbstractActor *;
using PosixActorPtr = PosixActor *;

}  // namespace message_infrastructure

#endif  // ABSTRACT_ACTOR_H_
