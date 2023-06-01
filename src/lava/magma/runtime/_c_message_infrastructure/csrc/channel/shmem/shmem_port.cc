// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <channel/shmem/shmem_port.h>
#include <core/utils.h>
#include <core/message_infrastructure_logging.h>
#include <semaphore.h>
#include <unistd.h>
#include <thread>  // NOLINT
#include <mutex>  // NOLINT
#include <memory>
#include <string>
#include <cassert>
#include <cstring>
#include <cstdlib>

namespace message_infrastructure {

namespace {

void MetaDataPtrFromPointer(const MetaDataPtr &ptr, void *p, int nbytes) {
  std::memcpy(ptr.get(), p, sizeof(MetaData));
  int len = ptr->elsize * ptr->total_size;
  if (len > nbytes) {
    LAVA_LOG_ERR("Recv %d data but max support %d length\n", len, nbytes);
    len = nbytes;
  }
  LAVA_DEBUG(LOG_SMMP, "data len: %d, nbytes: %d\n", len, nbytes);
  ptr->mdata = std::calloc(len, 1);
  if (ptr->mdata == nullptr) {
    LAVA_LOG_ERR("alloc failed, errno: %d\n", errno);
  }
  LAVA_DEBUG(LOG_SMMP, "memory allocates: %p\n", ptr->mdata);
  std::memcpy(ptr->mdata,
    reinterpret_cast<char *>(p) + sizeof(MetaData), len);
  LAVA_DEBUG(LOG_SMMP, "Metadata created\n");
}

}  // namespace

template<>
void RecvQueue<MetaDataPtr>::FreeData(MetaDataPtr data) {
  free(data->mdata);
}

ShmemSendPort::ShmemSendPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes)
  : AbstractSendPort(name, size, nbytes), shm_(shm), done_(false)
{}

void ShmemSendPort::Start() {
  shm_->Start();
}

void ShmemSendPort::Send(DataPtr metadata) {
  auto mdata = reinterpret_cast<MetaData*>(metadata.get());
  int len = mdata->elsize * mdata->total_size;
  if (len > nbytes_ - sizeof(MetaData)) {
    LAVA_LOG_ERR("Send data too large\n");
  }
  shm_->Store([len, &metadata](void* data){
    char* cptr = reinterpret_cast<char*>(data);
    std::memcpy(cptr, metadata.get(), sizeof(MetaData));
    cptr += sizeof(MetaData);
    std::memcpy(cptr,
                reinterpret_cast<MetaData*>(metadata.get())->mdata,
                len);
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
  recv_queue_ = std::make_shared<RecvQueue<MetaDataPtr>>(name_, size_);
}

ShmemRecvPort::~ShmemRecvPort() {
}

void ShmemRecvPort::Start() {
  recv_queue_thread_ = std::thread(
    &message_infrastructure::ShmemRecvPort::QueueRecv, this);
}

void ShmemRecvPort::QueueRecv() {
  while (!done_.load()) {
    bool ret = false;
    if (this->recv_queue_->AvailableCount() > 0) {
      bool not_empty = recv_queue_->Probe();
      ret = shm_->Load([this, &not_empty](void* data){
        MetaDataPtr metadata_res = std::make_shared<MetaData>();
        MetaDataPtrFromPointer(metadata_res, data,
                               nbytes_ - sizeof(MetaData));
        this->recv_queue_->Push(metadata_res);
      });
    }
    if (!ret) {
      helper::Sleep();
    }
  }
}

bool ShmemRecvPort::Probe() {
  return recv_queue_->Probe();
}

MetaDataPtr ShmemRecvPort::Recv() {
  return recv_queue_->Pop(true);
}

void ShmemRecvPort::Join() {
  if (!done_) {
    done_ = true;
    if (recv_queue_thread_.joinable())
      recv_queue_thread_.join();
    recv_queue_->Stop();
  }
}

MetaDataPtr ShmemRecvPort::Peek() {
  MetaDataPtr metadata_res = recv_queue_->Front();
  int mem_size = (nbytes_ - sizeof(MetaData) + 7) & (~0x7);
  void * ptr = std::calloc(mem_size, 1);
  if (ptr == nullptr) {
    LAVA_LOG_ERR("alloc failed, errno: %d\n", errno);
  }
  LAVA_DEBUG(LOG_SMMP, "memory allocates: %p\n", ptr);
  // memcpy to avoid double free
  // or maintain a address:refcount map
  std::memcpy(ptr, metadata_res->mdata, mem_size);
  MetaDataPtr metadata = std::make_shared<MetaData>();
  std::memcpy(metadata.get(), metadata_res.get(), sizeof(MetaData));
  metadata->mdata = ptr;
  return metadata;
}

ShmemBlockRecvPort::ShmemBlockRecvPort(const std::string &name,
  SharedMemoryPtr shm, const size_t &nbytes)
  : AbstractRecvPort(name, 1, nbytes), shm_(shm)
{}

MetaDataPtr ShmemBlockRecvPort::Recv() {
  MetaDataPtr metadata_res = std::make_shared<MetaData>();
  shm_->BlockLoad([&metadata_res, this](void* data){
    MetaDataPtrFromPointer(metadata_res, data,
                           nbytes_ - sizeof(MetaData));
  });
  return metadata_res;
}

MetaDataPtr ShmemBlockRecvPort::Peek() {
  MetaDataPtr metadata_res = std::make_shared<MetaData>();
  shm_->Read([&metadata_res, this](void* data){
    MetaDataPtrFromPointer(metadata_res, data,
                           nbytes_ - sizeof(MetaData));
  });
  return metadata_res;
}

bool ShmemBlockRecvPort::Probe() {
  return shm_->TryProbe();
}

}  // namespace message_infrastructure
