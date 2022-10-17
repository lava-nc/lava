// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHMEM_PORT_H_
#define SHMEM_PORT_H_

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>  // NOLINT

#include <message_infrastructure/csrc/core/abstract_port.h>
#include <message_infrastructure/csrc/channel/shmem/shm.h>

namespace message_infrastructure {

using ThreadPtr = std::shared_ptr<std::thread>;

class ShmemSendPort final : public AbstractSendPort {
 public:
  ShmemSendPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes);
  void Start();
  void Send(MetaDataPtr metadata);
  void Join();
  bool Probe();

 private:
  SharedMemoryPtr shm_ = nullptr;
  int idx_ = 0;
  std::atomic_bool done_;
  ThreadPtr ack_callback_thread_ = nullptr;
};

using ShmemSendPortPtr = std::shared_ptr<ShmemSendPort>;

class ShmemRecvQueue {
 public:
  ShmemRecvQueue(const std::string& name,
                 const size_t &size,
                 const size_t &nbytes);
  ~ShmemRecvQueue();
  void Push(void* src);
  void* Pop(bool block);
  void* Front();
  bool Probe();
  bool Empty();
  int AvailableCount();
  void Free();
  void Stop();

 private:
  std::string name_;
  size_t size_;
  size_t nbytes_;
  std::vector<void *> array_;
  std::atomic<uint32_t> read_index_;
  std::atomic<uint32_t> write_index_;
  std::atomic_bool done_;
};

using ShmemRecvQueuePtr = std::shared_ptr<ShmemRecvQueue>;

class ShmemRecvPort final : public AbstractRecvPort {
 public:
  ShmemRecvPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes);
  void Start();
  bool Probe();
  MetaDataPtr Recv();
  void Join();
  MetaDataPtr Peek();
  void QueueRecv();

 private:
  SharedMemoryPtr shm_ = nullptr;
  int idx_ = 0;
  std::atomic_bool done_;
  ShmemRecvQueuePtr queue_ = nullptr;
  ThreadPtr recv_queue_thread_ = nullptr;
};

using ShmemRecvPortPtr = std::shared_ptr<ShmemRecvPort>;

}  // namespace message_infrastructure

#endif  // SHMEM_PORT_H_
