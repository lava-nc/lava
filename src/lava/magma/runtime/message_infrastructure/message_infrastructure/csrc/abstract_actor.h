// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_ACTOR_H_
#define ABSTRACT_ACTOR_H_

#include <functional>

namespace message_infrastructure {

enum ActorStatus {
  StatsError = -1,
  StatsRuning = 0,
  StatsStopped = 1,
  StatsPaused = 2
};

class AbstractActor {
 public:
  virtual int GetPid() = 0;
  virtual int ForceStop() = 0;
  int pid_;
};

class PosixActor : public AbstractActor {
 public:
  explicit PosixActor(int pid, std::function<void()> target_fn) {
    this->pid_ = pid;
    this->target_fn_ = target_fn;
    this->status_ = StatsRuning;
  }
  int GetPid() {
    return this->pid_;
  }
  int Wait();
  int ForceStop();
  int ReStart();
  int GetStatus() {
    return this->status_;
  }
  // int Trace();
 private:
  std::function<void()> target_fn_ = NULL;
  int status_ = StatsStopped;
};

using ActorPtr = AbstractActor *;
using PosixActorPtr = PosixActor *;

}  // namespace message_infrastructure

#endif  // ABSTRACT_ACTOR_H_
