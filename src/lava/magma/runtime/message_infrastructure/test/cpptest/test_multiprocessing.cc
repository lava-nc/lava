// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/core/multiprocessing.h>
#include <message_infrastructure/csrc/core/abstract_actor.h>
#include <gtest/gtest.h>
#include <iostream>

namespace message_infrastructure {

class Builder {
 public:
    void Build(int i);
};

void Builder::Build(int i) {
  std::cout << "Builder running build " << i << std::endl;
  std::cout << "Build " << i << "... Sleeping for 3s" << std::endl;
  sleep(3);
  std::cout << "Build " << i << "... Builder complete" << std::endl;
}

void TargetFunction(Builder builder, int idx, AbstractActor* actor_ptr) {
  std::cout << "Target Function running... ID " << idx << std::endl;
  actor_ptr->SetStatus(ActorStatus::StatusStopped);
  builder.Build(idx);
}

TEST(TestMultiprocessing, MultiprocessingSpawn) {
  // Spawns an actor
  // Checks that actor is spawned successfully
  GTEST_SKIP();
  Builder *builder = new Builder();
  MultiProcessing mp;

  AbstractActor::TargetFn target_fn;

  for (int i = 0; i < 1; i++) {
    std::cout << "Loop " << i << std::endl;
    auto bound_fn = std::bind(&TargetFunction,
                              (*builder),
                              i,
                              std::placeholders::_1);
    target_fn = bound_fn;
    int return_value = mp.BuildActor(bound_fn);
    std::cout << "Return Value --> " << return_value << std::endl;
  }

  std::vector<AbstractActor::ActorPtr>& actorList = mp.GetActors();
  std::cout << "Actor List Length --> " << actorList.size() << std::endl;

  // Stop any currently running actors
  mp.Stop();
  mp.Cleanup(true);
}

TEST(TestMultiprocessing, ActorStop) {
  GTEST_SKIP();
  // Force stops all running actors
  // Checks that actor status returns 1 (StatusStopped)
  MultiProcessing mp;
  Builder *builder = new Builder();
  AbstractActor::TargetFn target_fn;

  for (int i = 0; i < 5; i++) {
    std::cout << "Loop " << i << std::endl;
    auto bound_fn = std::bind(&TargetFunction,
                              (*builder),
                              i,
                              std::placeholders::_1);
    target_fn = bound_fn;
    int return_value = mp.BuildActor(bound_fn);
    std::cout << "Return Value --> " << return_value << std::endl;
  }

  sleep(1);

  std::vector<AbstractActor::ActorPtr>& actorList = mp.GetActors();
  std::cout << "Actor List Length --> " << actorList.size() << std::endl;
  mp.Stop();
  mp.Cleanup(true);
}

}  // namespace message_infrastructure
