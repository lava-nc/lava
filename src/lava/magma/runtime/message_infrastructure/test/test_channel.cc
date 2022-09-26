// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <iostream>

#include <gtest/gtest.h>
#include <multiprocessing.h>
#include <abstract_actor.h>

using namespace message_infrastructure;

class Builder {
  public:
    void Build() {};
};

void TargetFunction(Builder builder, AbstractActor* actor_ptr) {
  std::cout << "Target Function running..." << std::endl;
  actor_ptr->SetStatus(ActorStatus::StatusStopped);
  builder.Build();
}

void SendPort() {

}

void RecvPort() {
  
}

TEST(TestSharedMemory, SharedMemSendReceive) {
  // Creates a pair of send and receive ports
  // TODO: Define success criteria
  MultiProcessing mp;
  Builder *builder_send = new Builder();
  Builder *builder_recv = new Builder();

  AbstractActor::TargetFn target_fn;
  
  // Stop any currently running actors
  mp.Stop(true);
}
