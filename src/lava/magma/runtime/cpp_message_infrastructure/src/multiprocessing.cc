// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "multiprocessing.h"

#include <sys/wait.h>
#include <unistd.h>
#include <iostream>

namespace message_infrastrature {

#define CPP_INFO "[CPP_INFO] "

void MultiProcessing::BuildActor(std::function<void()> target_fn) {
  pid_t pid = fork();

  if (pid < 0) {
    std::cout << CPP_INFO << "cannot allocate pid\n";
  }

  if (pid > 0) {
    std::cout << CPP_INFO << "parent, create actor\n";
    ActorPtr actor = new PosixActor(pid);
    actors_.push_back(actor);
  }

  if (pid == 0) {
    std::cout << CPP_INFO << "child, new process\n";
    target_fn();
    exit(0);
  }

}

void MultiProcessing::Stop() {
}

void MultiProcessing::CheckActor() {
  for(auto actor : actors_){
    std::cout << CPP_INFO << actor->pid_ << std::endl;
  }
}

} // namespace message_infrastrature
