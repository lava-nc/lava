// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ACTOR_POSIX_ACTOR_H_
#define ACTOR_POSIX_ACTOR_H_

#include <message_infrastructure/csrc/core/abstract_actor.h>

namespace message_infrastructure {

class PosixActor final : public AbstractActor {
 public:
  using AbstractActor::AbstractActor;
  ~PosixActor() override {}
  int GetPid();
  int Wait();
  int ForceStop();
  int Create();
};

using PosixActorPtr = PosixActor *;

}  // namespace message_infrastructure

#endif  // ACTOR_POSIX_ACTOR_H_
