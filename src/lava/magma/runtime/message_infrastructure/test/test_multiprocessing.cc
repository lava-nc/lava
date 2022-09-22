// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <iostream>

#include <gtest/gtest.h>
#include <multiprocessing.h>

TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

using namespace message_infrastructure;

class Builder {
  public:
    void Build(int i);
};

void Builder::Build(int i) {
  std::cout << "Builder running build " << i << std::endl;
  std::cout << "Build " << i << "... Sleeping for 10s" << std::endl;
  sleep(10);
  std::cout << "Build " << i << "... Builder complete" << std::endl;
}

AbstractActor::TargetFn TargetFunction(Builder builder, int idx) {
  std::cout << "Target Function running" << std::endl;
  builder.Build(idx);
}

TEST(TestMultiprocessing, MultiprocessingSpawn) {
  // Spawns an actor
  // Checks that actor is spawned successfully
  Builder *builder = new Builder();
  MultiProcessing mp;

  for (int i = 0; i < 5; i++) {
    std::cout << "Loop " << i << std::endl;
    auto bound_fn = std::bind(TargetFunction, std::placeholders::_1, i);
    mp.BuildActor(bound_fn(*builder));
  }
}

TEST(TestMultiprocessing, MultiprocessingShutdown) {
  // Spawns an actor and sends a stop signal
  // Checks that actor is stopped successfully
  GTEST_SKIP() << "Skipping MultiprocessingShutdown";
}

TEST(TestMultiprocessing, ActorForceStop) {
  // Force stops all running actors
  // Checks that actor status returns 1 (StatusStopped)
  GTEST_SKIP() << "Skipping ActorForceStop";
  MultiProcessing mp;
  std::vector<AbstractActor::ActorPtr>& actorList = mp.GetActors();
  for (auto actor : actorList){
    actor->ForceStop();
    int actorStatus = actor->GetStatus();
    EXPECT_EQ(actorStatus, 1);
  }
}

TEST(TestMultiprocessing, ActorRunning) {
  // Checks that acto status returns 0 (StatusRuning)
  // std::vector<ActorPtr>& actor_list = *Multiprocessing()
  MultiProcessing mp;
  std::vector<AbstractActor::ActorPtr>& actorList = mp.GetActors();
  for (auto actor : actorList){
    int actorStatus = actor->GetStatus();
    EXPECT_EQ(actorStatus, 0);
  }
}
