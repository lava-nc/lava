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

ShmemChannel::ShmemChannel(const SharedMemManager &smm,
                           const std::string &src_name,
                           const std::string &dst_name,
                           const size_t &size,
                           const size_t &nbytes) {
  smm_ = smm;
  int shmid = smm_.AllocSharedMemoryWithName(src_name, nbytes * size);
  SharedMemory shm(shmid);

  std::string req_name = src_name + "_req";
  std::string ack_name = src_name + "_ack";

  req_ = sem_open(req_name.c_str(), CREAT_FLAG, ACC_MODE, 0);
  ack_ = sem_open(ack_name.c_str(), CREAT_FLAG, ACC_MODE, 0);

  if (req_ == SEM_FAILED || req_ == SEM_FAILED) {
    printf("Create sem fail for channel %s\n", src_name.c_str());
    exit(-1);
  }

  AbstractSendPortPtr send_port = std::make_shared<ShmemSendPort>(src_name, shm, size, nbytes);
  AbstractRecvPortPtr recv_port = std::make_shared<ShmemRecvPort>(dst_name, shm, size, nbytes);
  
  send_port_proxy_ = std::make_shared<SendPortProxy>(
      ChannelType::SHMEMCHANNEL,
      send_port);
  recv_port_proxy_ = std::make_shared<RecvPortProxy>(
      ChannelType::SHMEMCHANNEL,
      recv_port);
}

SendPortProxyPtr ShmemChannel::GetSendPort() {
  printf("Get send_port.\n");
  return this->send_port_proxy_;
}

RecvPortProxyPtr ShmemChannel::GetRecvPort() {
  printf("Get recv_port.\n");
  return this->recv_port_proxy_;
}

ShmemChannel::~ShmemChannel() {
  sem_destroy(req_);
  sem_destroy(ack_);
}

} // namespace message_infrastructure
