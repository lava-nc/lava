// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHMEM_CHANNEL_H_
#define SHMEM_CHANNEL_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <memory>
#include <string>

#include "abstract_channel.h"
#include "shm.h"

namespace message_infrastructure {

class ShmemChannel : public AbstractChannel {
 public:
  ShmemChannel(SharedMemoryPtr shm,
               const std::string &src_name,
               const std::string &dst_name,
               const ssize_t &shape,
               const pybind11::dtype &dtype,
               const size_t &size);
  std::shared_ptr<AbstractSendPort> GetSrcPort() {
    return src_port_;
  }
  std::shared_ptr<AbstractRecvPort> GetDstPort() {
    return dst_port_;
  }

 private:
  SharedMemoryPtr shm_ = NULL;
  sem_t *req_ = NULL;
  sem_t *ack_ = NULL;
};

using ShmemChannelPtr = ShmemChannel *;

}  // namespace message_infrastructure

#endif  // SHMEM_CHANNEL_H_

