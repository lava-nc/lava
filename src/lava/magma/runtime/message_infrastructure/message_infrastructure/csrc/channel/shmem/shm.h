// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_SHMEM_SHM_H_
#define CHANNEL_SHMEM_SHM_H_

#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <semaphore.h>
#include <memory>
#include <map>
#include <set>
#include <string>
#include <atomic>
#include <functional>
#include <cstdlib>
#include <ctime>

namespace message_infrastructure {

#define SHM_FLAG O_RDWR | O_CREAT
#define SHM_MODE S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH

using HandleFn = std::function<void(void *)>;

class SharedMemory {
 public:
  SharedMemory() {}
  SharedMemory(const size_t &mem_size, void* mmap, const int &key);
  SharedMemory(const size_t &mem_size, void* mmap);
  ~SharedMemory();
  void Start();
  bool Load(HandleFn consume_fn);
  void BlockLoad(HandleFn consume_fn);
  void Read(HandleFn consume_fn);
  void Store(HandleFn store_fn);
  void Close();
  bool TryProbe();
  void InitSemaphore();
  int GetDataElem(int offset);
  std::string GetReq();
  std::string GetAck();

 private:
  size_t size_;
  std::string req_name_ = "req";
  std::string ack_name_ = "ack";
  sem_t *req_;
  sem_t *ack_;
  void *data_ = NULL;
};

class RwSharedMemory {
 public:
  RwSharedMemory(const size_t &mem_size, void* mmap, const int &key);
  ~RwSharedMemory();
  void InitSemaphore();
  void Start();
  void Handle(HandleFn handle_fn);
  void Close();

 private:
  size_t size_;
  std::string sem_name_ = "sem";
  sem_t *sem_;
  void *data_;
};

using SharedMemoryPtr = std::shared_ptr<SharedMemory>;
using RwSharedMemoryPtr = std::shared_ptr<RwSharedMemory>;

class SharedMemManager {
 public:
  template<typename T>
  std::shared_ptr<T> AllocChannelSharedMemory(const size_t &mem_size) {
    int random = std::rand();
    std::string str = shm_str_ + std::to_string(random);
    int shmfd = shm_open(str.c_str(), SHM_FLAG, SHM_MODE);
    LAVA_DEBUG(LOG_SMMP, "Shm fd and name open: %s %d\n",
               str.c_str(), shmfd);
    if (shmfd == -1) {
      LAVA_LOG_ERR("Create shared memory object failed.\n");
      exit(-1);
    }
    int err = ftruncate(shmfd, mem_size);
    if (err == -1) {
      LAVA_LOG_ERR("Resize shared memory segment failed.\n");
      exit(-1);
    }
    shm_fd_strs_.insert({shmfd, str});
    void *mmap_address = mmap(NULL, mem_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, shmfd, 0);
    if (mmap_address == reinterpret_cast<void*>(-1)) {
      LAVA_LOG_ERR("Get shmem address error, errno: %d\n", errno);
      LAVA_DUMP(1, "size: %ld, shmfd_: %d\n", mem_size, shmfd);
    }
    std::shared_ptr<T> shm =
      std::make_shared<T>(mem_size, mmap_address, random);
    sem_strs_.insert(shm->GetReq());
    sem_strs_.insert(shm->GetAck());
    shm->InitSemaphore();
    return shm;
  }

  void DeleteAllSharedMemory();
  friend SharedMemManager &GetSharedMemManager();

 private:
  SharedMemManager() {
    std::srand(std::time(nullptr));
    alloc_pid_ = getpid();
  }
  std::map<int, std::string> shm_fd_strs_;
  std::set<std::string> sem_strs_;
  static SharedMemManager smm_;
  std::string shm_str_ = "shm";
  int alloc_pid_;
};

SharedMemManager& GetSharedMemManager();

using SharedMemManagerPtr = std::shared_ptr<SharedMemManager>;

}  // namespace message_infrastructure

#endif  // CHANNEL_SHMEM_SHM_H_
