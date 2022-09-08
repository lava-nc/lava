// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHMEM_PORT_H_
#define SHMEM_PORT_H_

#include <queue>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>  // NOLINT
#include <condition_variable>  // NOLINT
#include <thread>  // NOLINT

#include "abstract_port.h"
#include "shm.h"
#include "utils.h"

namespace message_infrastructure {

using ThreadPtr = std::shared_ptr<std::thread>;

class ShmemSendPort : public AbstractSendPort {
 public:
  ShmemSendPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes);
  std::string Name();
  size_t Size();
  void Start();
  int Probe();
  int Send(void* data);
  void Join();
  void Stop();
  int AckCallback();

  SharedMemoryPtr shm_ = NULL;
  int idx_ = 0;
  std::atomic_bool done_;
  void *array_ = NULL;
  sem_t *semaphore_ = NULL;
  void *observer = NULL;
  ThreadPtr ack_callback_thread_ = NULL;
};

class ShmemRecvQueue {
 public:
  ShmemRecvQueue(const std::string& name,
                 const size_t &size,
                 const size_t &nbytes);
  ~ShmemRecvQueue();
  void Push(void* src);
  void Pop();
  void* Front();
  void* FrontPop();
  bool Probe();
  bool Empty();
  void Free();

 private:
  std::string name_;
  size_t nbytes_;
  size_t size_;
  std::vector<void *> array_;
  std::vector<void *> drop_array_;
  std::atomic<uint32_t> read_index_;
  std::atomic<uint32_t> write_index_;
};

using ShmemRecvQueuePtr = std::shared_ptr<ShmemRecvQueue>;

class ShmemRecvPort : public AbstractRecvPort {
 public:
  ShmemRecvPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes);
  std::string Name();
  size_t Size();
  void Start();
  bool Probe();
  void* Recv();
  void Join();
  void* Peek();
  int ReqCallback();
  void QueueRecv();

  SharedMemoryPtr shm_ = NULL;
  int idx_ = 0;
  std::atomic_bool done_;
  void *array_ = NULL;
  void *observer = NULL;
  ShmemRecvQueuePtr queue_ = NULL;
  ThreadPtr req_callback_thread_ = NULL;
  ThreadPtr recv_queue_thread_ = NULL;
};

}  // namespace message_infrastructure

#endif  // SHMEM_PORT_H_
