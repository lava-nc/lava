// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHMEM_CHANNEL_H_
#define SHMEM_CHANNEL_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <semaphore.h>

#include <memory>
#include <string>

#include "abstract_channel.h"
#include "shm.h"
#include "utils.h"
#include "port_proxy.h"

namespace message_infrastructure {

class ShmemChannel : public AbstractChannel {
 public:
  ShmemChannel() {}
  ShmemChannel(const SharedMemManager &smm,
               const std::string &src_name,
               const std::string &dst_name,
               const size_t &size,
               const size_t &nbytes);
  ~ShmemChannel();
  SendPortProxyPtr GetSendPort();
  RecvPortProxyPtr GetRecvPort();
 private:
  SharedMemManager smm_;
  sem_t *req_ = NULL;
  sem_t *ack_ = NULL;
  SendPortProxyPtr send_port_proxy_ = NULL;
  RecvPortProxyPtr recv_port_proxy_ = NULL;
  size_t size_;
  size_t nbytes_;
  size_t name_;
};

std::shared_ptr<ShmemChannel> GetShmemChannel(const SharedMemManager &smm,
                              const size_t &size,
                              const size_t &nbytes,
                              const std::string &name) {
  printf("Generate shmem_channel.\n");
  return (std::make_shared<ShmemChannel>(smm,
                                         name,
                                         name,
                                         size,
                                         nbytes));
}

}  // namespace message_infrastructure

#endif  // SHMEM_CHANNEL_H_
