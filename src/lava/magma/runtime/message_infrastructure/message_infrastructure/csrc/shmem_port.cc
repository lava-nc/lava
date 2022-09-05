// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pthread.h>
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
#include <memory.h>

#include "shmem_port.h"
#include "shm.h"
#include "utils.h"

namespace message_infrastructure {

ShmemSendPort::ShmemSendPort(const std::string &name,
                SharedMemory shm,
                const size_t &size,
                const size_t &nbytes) {
  name_ = name;
  shm_ = shm;
  nbytes_ = nbytes;
  size_ = size;
  
  void *ptr = shmat(shm.GetShmid(), NULL, 0);

  for(int i = 0; i < size_; i++) {
    array_.push_back(ptr + nbytes_ * i);
  }
}

int ShmemSendPort::Start() {
  return 0;
}

int ShmemSendPort::Probe() {
}

int ShmemSendPort::Send(void* data) {
  return 0;
}

int ShmemSendPort::Join() {
  return 0;
}

int ShmemSendPort::AckCallback() {
  return 0;
}

std::string ShmemSendPort::Name() {
  return name_;
}

size_t ShmemSendPort::Size() {
  return size_;
}

ShmemRecvPort::ShmemRecvPort(const std::string &name,
                SharedMemory shm,
                const size_t &size,
                const size_t &nbytes) {
  name_ = name;
  shm_ = shm;
  nbytes_ = nbytes;
  size_ = size;
  
  void *ptr = shmat(shm.GetShmid(), NULL, 0);

  for(int i = 0; i < size_; i++) {
    array_.push_back(ptr + nbytes_ * i);
  }
}

int ShmemRecvPort::Start() {
  return 0;
}

bool ShmemRecvPort::Probe() {
  return true;
}

void* ShmemRecvPort::Recv() {
  return NULL;
}

int ShmemRecvPort::Join() {
  return 0;
}

void* ShmemRecvPort::Peek() {
  return NULL;
}

int ShmemRecvPort::ReqCallback() {
  return 0;
}

} // namespace message_infrastructure
