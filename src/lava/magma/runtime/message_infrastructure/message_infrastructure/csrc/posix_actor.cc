// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_actor.h"
#include "message_infrastructure_logging.h"

#include <sys/wait.h>
#include <sys/types.h>

namespace message_infrastructure {

int PosixActor::Stop() {
  int status;
  // TODO: Add the options can be as args of the function
  int options = 0;
  int ret = waitpid(this->pid_, &status, options);

  if (ret < 0) {
    LAVA_LOG_ERR("process %d waitpid error\n", this->pid_);
    return -1;
  }

  // Check the status
  return 0;

}

int PosixActor::Wait() {
  return 0;
}

}