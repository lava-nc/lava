// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef INCLUDE_SHMEM_CHANNEL_H_
#define INCLUDE_SHMEM_CHANNEL_H_

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "abstract_channel.h"
#include "shm.h"

namespace message_infrastructure {

class ShmemChannel : public AbstractChannel {
 public:
  ShmemChannel(SharedMemory *shm,
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
  SharedMemory *shm_ = NULL;
  sem_t *req_ = NULL;
  sem_t *ack_ = NULL;
};

template <class T>
ShmemChannel* GetShmemChannel(SharedMemory *shm,
                              pybind11::array_t<T> &data,
                              const size_t &size,
                              const std::string &name = "test_channel") {
  return (new ShmemChannel(shm, name, name, data.shape(), data.dtype(), size));
}

} // namespace message_infrastructure

#endif
