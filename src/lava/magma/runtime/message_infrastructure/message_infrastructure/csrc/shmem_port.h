// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHMEM_PORT_H_
#define SHMEM_PORT_H_

#include <thread>  // NOLINT [build/c++11]
#include <queue>
#include <string>

#include "abstract_port.h"
#include "shm.h"
#include "utils.h"

namespace message_infrastructure {

template<class T>
class ShmemRecvQueue {
 public:
  T get(bool block = true, time_t timeout = 0, bool peek = false);
 private:
  std::queue<T> queue_;
};
template<class T>
class ShmemSendPort : public AbstractSendPort {
 public:
  ShmemSendPort(const std::string &name,
                SharedMemoryPtr shm,
                Proto *proto,
                const size_t &size,
                sem_t *req,
                sem_t *ack);
  std::string Name();
  pybind11::dtype Dtype();
  ssize_t* Shape();
  size_t Size();
  int Start();
  int Probe();
  int Send();
  int Join();
  int AckCallback();

  SharedMemoryPtr shm_ = NULL;
  sem_t *req_ = NULL;
  sem_t *ack_ = NULL;
  int idx_ = 0;
  bool done_ = false;
  void *array_ = NULL;
  sem_t *semaphore_ = NULL;
  void *observer = NULL;
  std::thread *thread_ = NULL;
};
template<class T>
class ShmemRecvPort : public AbstractRecvPort {
 public:
  ShmemRecvPort(const std::string &name,
                SharedMemoryPtr shm,
                Proto *proto,
                const size_t &size,
                sem_t *req,
                sem_t *ack);
  std::string Name();
  pybind11::dtype Dtype();
  ssize_t* Shape();
  size_t Size();
  int Start();
  int Probe();
  int Recv();
  int Join();
  int Peek();
  int ReqCallback();

  SharedMemoryPtr shm_ = NULL;
  sem_t *req_ = NULL;
  sem_t *ack_ = NULL;
  int idx_ = 0;
  bool done_ = false;
  void *array_ = NULL;
  void *observer = NULL;
  std::thread *thread_ = NULL;
  ShmemRecvQueue<T> *queue = NULL;
};

}  // namespace message_infrastructure

#endif  // SHMEM_PORT_H_
