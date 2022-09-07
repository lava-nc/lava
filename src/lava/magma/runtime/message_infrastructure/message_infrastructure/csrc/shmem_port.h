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
#include <mutex>  // NOLINT
#include <condition_variable>  // NOLINT

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
  void *array_ = NULL;
  sem_t *semaphore_ = NULL;
  void *observer = NULL;
};

class ShmemRecvQueue {
 public:
  void Init(const size_t &capacity, const size_t &nbytes);
  void Push(void* src);
  void Pop();
  void* Front();
  void* FrontPop();
  bool Probe();

 private:
  std::mutex lock_;
  std::condition_variable cond_;
  size_t nbytes_;
  size_t capacity_;
  std::queue<void *> queue_;
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
  void QueueRecv();

  SharedMemory shm_;
  int idx_ = 0;
  std::atomic_bool done_;
  void *array_ = NULL;
  void *observer = NULL;
  ShmemRecvQueue queue_;
  void *dst;
};

}  // namespace message_infrastructure

#endif  // SHMEM_PORT_H_
