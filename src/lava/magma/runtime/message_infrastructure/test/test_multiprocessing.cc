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

std::function<int(ActorPtr)> TargetFunction(Builder builder, int idx) {
  std::cout << "Target Function running" << std::endl;
  builder.Build(idx);
}

TEST(TestMultiprocessing, MultiprocessingSpawn) {
  // Spawns an actor
  // Checks that actor is spawned successfully
  Builder *builder = new Builder();

  for (int i = 0; i < 5; i++) {
    auto bound_fn = std::bind(TargetFunction, std::placeholders::_1, i);
    // TODO: Add multiprocessing BuildActor
    // bound_fn(*builder);
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
}

TEST(TestMultiprocessing, ActorRunning) {
  // Checks that acto status returns 0 (StatusRuning)
  // std::vector<ActorPtr>& actor_list = *Multiprocessing()
}

TEST(TestMultiprocessing, ActorStop) {
  // Stops all running actors
  // Checks that actor status returns 1 (StatusStopped)
}
