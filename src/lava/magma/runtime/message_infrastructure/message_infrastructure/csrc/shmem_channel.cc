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
  unsigned long shmem_size = nbytes + sizeof(MetaData);

  shm_ = GetSharedMemManager().AllocChannelSharedMemory<SharedMemory>(shmem_size);

  send_port_ = std::make_shared<ShmemSendPort>(src_name, shm_, size, shmem_size);
  recv_port_ = std::make_shared<ShmemRecvPort>(dst_name, shm_, size, shmem_size);
}

AbstractSendPortPtr ShmemChannel::GetSendPort() {
  return send_port_;
}

AbstractRecvPortPtr ShmemChannel::GetRecvPort() {
  return recv_port_;
}

std::shared_ptr<ShmemChannel> GetShmemChannel(const size_t &size,
                              const size_t &nbytes,
                              const std::string &name) {
  return (std::make_shared<ShmemChannel>(name,
                                         name,
                                         size,
                                         nbytes));
}
}  // namespace message_infrastructure
