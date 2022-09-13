// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <string>

#include "shmem_channel.h"
#include "shmem_port.h"
#include "port_proxy.h"
#include "utils.h"
#include "shm.h"

namespace message_infrastructure {

ShmemChannel::ShmemChannel(const std::string &src_name,
                           const std::string &dst_name,
                           const size_t &size,
                           const size_t &nbytes) {
  shm_ = GetSharedMemManager().AllocChannelSharedMemory(nbytes);

  send_port_ = std::make_shared<ShmemSendPort>(src_name, shm_, size, nbytes + offsetof(MetaData, mdata));
  recv_port_ = std::make_shared<ShmemRecvPort>(dst_name, shm_, size, nbytes + offsetof(MetaData, mdata));
}

AbstractSendPortPtr ShmemChannel::GetSendPort() {
  printf("Get shmem send_port.\n");
  return send_port_;
}

AbstractRecvPortPtr ShmemChannel::GetRecvPort() {
  printf("Get shmem recv_port.\n");
  return recv_port_;
}

std::shared_ptr<ShmemChannel> GetShmemChannel(const size_t &size,
                              const size_t &nbytes,
                              const std::string &name) {
  printf("Generate shmem_channel.\n");
  return (std::make_shared<ShmemChannel>(name,
                                         name,
                                         size,
                                         nbytes));
}
}  // namespace message_infrastructure
