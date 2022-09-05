// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHMEM_PORT_H_
#define SHMEM_PORT_H_

#include <pthread.h>
#include <queue>
#include <string>
#include <vector>
#include <atomic> 

#include "abstract_port.h"
#include "shm.h"
#include "utils.h"

namespace message_infrastructure {

class ShmemSendPort : public AbstractSendPort {
 public:
  ShmemSendPort(const std::string &name,
                SharedMemory shm,
                const size_t &size,
                const size_t &nbytes);
  std::string Name();
  size_t Size();
  int Start();
  int Probe();
  int Send(void* data);
  int Join();
  int AckCallback();

  SharedMemory shm_;
  int idx_ = 0;
  std::atomic_bool done_;
  std::vector<void *> array_;
  sem_t *semaphore_ = NULL;
  void *observer = NULL;
};
class ShmemRecvPort : public AbstractRecvPort {
 public:
  ShmemRecvPort(const std::string &name,
                SharedMemory shm,
                const size_t &size,
                const size_t &nbytes);
  std::string Name();
  size_t Size();
  int Start();
  bool Probe();
  void* Recv();
  int Join();
  void* Peek();
  int ReqCallback();

  SharedMemory shm_;
  int idx_ = 0;
  std::atomic_bool done_;
  std::vector<void *> array_;
  void *observer = NULL;
};

}  // namespace message_infrastructure

#endif  // SHMEM_PORT_H_
