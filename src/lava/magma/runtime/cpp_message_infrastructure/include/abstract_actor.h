// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTACT_ACTOR_H_
#define ABSTACT_ACTOR_H_

namespace message_infrastructure {

class AbstractActor {
 public:
  virtual int GetPid() = 0;
  virtual int Stop() = 0;
  virtual int Pause() = 0;

  int pid_;
};

class PosixActor : public AbstractActor {
 public:
  explicit PosixActor(int pid){
    this->pid_ = pid;
  }
  int GetPid(){
    return this->pid_;
  };
  int Stop(){
    return 0;
  };
  int Pause(){
    return 0;
  };
};

using ActorPtr = AbstractActor *;
using PosixActorPtr = PosixActor *;

} // namespace message_infrastructure

#endif