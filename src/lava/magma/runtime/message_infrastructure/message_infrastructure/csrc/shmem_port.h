// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHMEM_PORT_H_
#define SHMEM_PORT_H_

#include <pybind11/numpy.h>

#include <thread>  // NOLINT [build/c++11]
#include <queue>
#include <string>
#include <vector>

#include "abstract_port.h"
#include "shm.h"
#include "utils.h"

namespace message_infrastructure {

template<class T>
class ShmemSendPort : public AbstractSendPort {
 public:
  ShmemSendPort(const std::string &name,
                const SharedMemory &shm,
                const size_t &size,
                const size_t &nbytes);
  std::string Name();
  // pybind11::dtype Dtype();
  // ssize_t* Shape();
  size_t Size();
  int Start();
  int Probe();
  int Send(T* data);
  int Join();
  int AckCallback();

  SharedMemory &shm_;
  sem_t *req_ = NULL;
  sem_t *ack_ = NULL;
  int idx_ = 0;
  bool done_ = false;
  // std::vector<pybind11::array> array_;
  std::vector<T *> array_;
  // void *array_ = NULL;
  sem_t *semaphore_ = NULL;
  void *observer = NULL;
  std::thread *thread_ = NULL;
};
template<class T>
class ShmemRecvPort : public AbstractRecvPort {
 public:
  ShmemRecvPort(const std::string &name,
                const SharedMemory &shm,
                const size_t &size,
                const size_t &nbytes);
  std::string Name();
  // pybind11::dtype Dtype();
  // ssize_t* Shape();
  size_t Size();
  int Start();
  bool Probe();
  T* Recv();
  int Join();
  T* Peek();
  int ReqCallback();

  SharedMemory shm_;
  sem_t *req_ = NULL;
  sem_t *ack_ = NULL;
  int idx_ = 0;
  bool done_ = false;
  // std::vector<pybind11::array*> array_;
  std::vector<T *> array_;
  // void *array_ = NULL;
  void *observer = NULL;
  std::thread *thread_ = NULL;
  // ShmemRecvQueue<pybind11::array_t<T>> queue;
};

}  // namespace message_infrastructure

#endif  // SHMEM_PORT_H_
