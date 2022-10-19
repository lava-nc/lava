// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_MULTIPROCESSING_H_
#define CORE_MULTIPROCESSING_H_

#include <message_infrastructure/csrc/core/abstract_actor.h>
#include <vector>
#include <functional>

namespace message_infrastructure {

class MultiProcessing {
 public:
  ~MultiProcessing();
  void Stop(bool block);
  int BuildActor(AbstractActor::TargetFn target_fn);
  void CheckActor();
  std::vector<AbstractActor::ActorPtr>& GetActors();

 private:
  std::vector<AbstractActor::ActorPtr> actors_;
};

}  // namespace message_infrastructure

#endif  // CORE_MULTIPROCESSING_H_
