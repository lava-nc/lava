// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/shmem/shmem_port.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <semaphore.h>
#include <unistd.h>
#include <thread>  // NOLINT
#include <mutex>  // NOLINT
#include <memory>
#include <string>
#include <condition_variable>  // NOLINT
#include <cassert>
#include <cstring>

namespace message_infrastructure {

ShmemSendPort::ShmemSendPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes)
  : AbstractSendPort(name, size, nbytes), shm_(shm), done_(false)
{}

void ShmemSendPort::Start() {
  shm_->Start();
}

void ShmemSendPort::Send(MetaDataPtr metadata) {
  shm_->Store([this, &metadata](void* data){
    char* cptr = reinterpret_cast<char*>(data);
    std::memcpy(cptr, metadata.get(), sizeof(MetaData));
    cptr += sizeof(MetaData);
    std::memcpy(cptr, metadata->mdata, this->nbytes_ - sizeof(MetaData));
  });
}

bool ShmemSendPort::Probe() {
  return false;
}

void ShmemSendPort::Join() {
  done_ = true;
}

ShmemRecvPort::ShmemRecvPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes)
  : AbstractRecvPort(name, size, nbytes), shm_(shm), done_(false) {
  recvqueue = std::make_shared<RecvQueue<void*>>(name_, size_);
}
ShmemRecvPort::~ShmemRecvPort() {
  recvqueue->Free();
}
void ShmemRecvPort::Start() {
  recv_queue_thread_ = std::make_shared<std::thread>(
                       &message_infrastructure::ShmemRecvPort::QueueRecv, this);
}

void ShmemRecvPort::QueueRecv() {
  while (!done_.load()) {
    bool ret = false;
    if (this->recvqueue->AvailableCount() > 0) {
      ret = shm_->Load([this](void* data){
      void *ptr = malloc(this->nbytes_);
      std::memcpy(ptr, data, this->nbytes_);
        this->recvqueue->Push(ptr);
      });
    }
    if (!ret) {
      // sleep
      helper::Sleep();
    }
  }
}

bool ShmemRecvPort::Probe() {
  return recvqueue->Probe();
}

MetaDataPtr ShmemRecvPort::Recv() {
  char *cptr = reinterpret_cast<char *>(recvqueue->Pop(true));
  MetaDataPtr metadata_res = std::make_shared<MetaData>();
  std::memcpy(metadata_res.get(), cptr, sizeof(MetaData));
  metadata_res->mdata = reinterpret_cast<void*>(cptr + sizeof(MetaData));
  return metadata_res;
}

void ShmemRecvPort::Join() {
  if (!done_) {
    done_ = true;
    recv_queue_thread_->join();
    recvqueue->Stop();
  }
}

MetaDataPtr ShmemRecvPort::Peek() {
  char *cptr = reinterpret_cast<char *>(recvqueue->Front());
  MetaDataPtr metadata_res = std::make_shared<MetaData>();
  std::memcpy(metadata_res.get(), cptr, sizeof(MetaData));
  metadata_res->mdata = reinterpret_cast<void*>(cptr + sizeof(MetaData));
  return metadata_res;
}

}  // namespace message_infrastructure