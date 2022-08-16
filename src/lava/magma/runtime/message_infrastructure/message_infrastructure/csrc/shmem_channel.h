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
               const ssize_t* shape,
               const pybind11::dtype &dtype,
               const size_t &size,
               const size_t &nbytes);
  std::shared_ptr<AbstractSendPort> GetSendPort() {
    return this->send_port_;
  }
  std::shared_ptr<AbstractRecvPort> GetRecvPort() {
    return this->recv_port_;
  }

 private:
  SharedMemoryPtr shm_ = NULL;
  sem_t *req_ = NULL;
  sem_t *ack_ = NULL;
};

using ShmemChannelPtr = ShmemChannel *;

template <class T>
std::shared_ptr<ShmemChannel> GetShmemChannel(SharedMemoryPtr shm,
                              const pybind11::array_t<T> &data,
                              const size_t &size,
                              const size_t &nbytes,
                              const std::string &name = "test_channel") {
  return (std::make_shared<ShmemChannel>(shm,
                                         name,
                                         name,
                                         data.shape(),
                                         data.dtype(),
                                         size, nbytes));
}

}  // namespace message_infrastructure

#endif  // SHMEM_CHANNEL_H_

