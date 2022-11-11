// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_SHMEM_SHMEM_PORT_H_
#define CHANNEL_SHMEM_SHMEM_PORT_H_

#include <message_infrastructure/csrc/core/abstract_port.h>
#include <message_infrastructure/csrc/channel/shmem/shm.h>
#include <message_infrastructure/csrc/core/common.h>

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>  // NOLINT

namespace message_infrastructure {

using ThreadPtr = std::shared_ptr<std::thread>;

template class RecvQueue<MetaDataPtr>;

class ShmemSendPort final : public AbstractSendPort {
 public:
  ShmemSendPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes);
  void Start();
  void Send(MetaDataPtr metadata);
  void Send(GrpcMetaDataPtr metadata) {}
  void Join();
  bool Probe();

 private:
  SharedMemoryPtr shm_ = nullptr;
  std::atomic_bool done_;
  ThreadPtr ack_callback_thread_ = nullptr;
};

// Users should be allowed to copy port objects.
// Use std::shared_ptr.
using ShmemSendPortPtr = std::shared_ptr<ShmemSendPort>;

class ShmemBlockRecvPort final : public AbstractRecvPort {
 public:
  ShmemBlockRecvPort(const std::string &name,
                     SharedMemoryPtr shm,
                     const size_t &nbytes);
  ~ShmemBlockRecvPort() {}
  void Start() {}
  bool Probe();
  MetaDataPtr Recv();
  void Join() {}
  MetaDataPtr Peek();

 private:
  SharedMemoryPtr shm_ = nullptr;
};

class ShmemRecvPort final : public AbstractRecvPort {
 public:
  ShmemRecvPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes);
  ~ShmemRecvPort();
  void Start();
  bool Probe();
  MetaDataPtr Recv();
  void Join();
  MetaDataPtr Peek();
  void QueueRecv();

 private:
  SharedMemoryPtr shm_ = nullptr;
  std::atomic_bool done_;
  std::shared_ptr<RecvQueue<MetaDataPtr>> recv_queue_;
  ThreadPtr recv_queue_thread_ = nullptr;
};

// Users should be allowed to copy port objects.
// Use std::shared_ptr.
using ShmemRecvPortPtr = std::shared_ptr<ShmemRecvPort>;

}  // namespace message_infrastructure

#endif  // CHANNEL_SHMEM_SHMEM_PORT_H_
