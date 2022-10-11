// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef MULTIPROCESSING_H_
#define MULTIPROCESSING_H_

#include <vector>
#include <functional>

#include "abstract_actor.h"

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

#endif  // MULTIPROCESSING_H_
