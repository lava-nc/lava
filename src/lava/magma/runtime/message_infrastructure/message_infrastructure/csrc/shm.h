// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHM_H_
#define SHM_H_

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>

#include <memory>
#include <set>
#include <string>

namespace message_infrastructure {

class SharedMemory {
};

using SharedMemoryPtr = SharedMemory*;

class SharedMemManager {
 public:
  int AllocSharedMemory(const std::string &src_name, const size_t mem_size) {
    std::string shm_name = src_name + "_shm";

    int shmid = shm_open(shm_name.c_str(),
      O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);

    if (shmid == -1) {
      printf("Create shared memory object fail..\n");
      exit(-1);
    }

    int err_1 = ftruncate(shmid, mem_size);

    if (err_1 == -1) {
      printf("Resize shared memory segment fail..\n");
      exit(-1);
    }

    struct stat stat_shmem;

    err_1 = fstat(shmid, &stat_shmem);
    if (err_1 == -1) {
      printf("Detect shared memory segment size fail..\n");
      exit(-1);
    }

    shms_.insert(shm_name);
    return shmid;
  }

  int AllocSharedMemory(const size_t mem_size) {
    int shmid = shmget(key_, mem_size, 0644|IPC_CREAT);
    if (shmid < 0)
      return -1;

    shmids_.insert(shmid);
    key_++;
    return shmid;
  }

  int DeleteSharedMemory(const int shmid) {
    // Release specific shared memory
    int result = -1;
    if (shmids_.find(shmid) != shmids_.end()) {
      result = shmctl(shmid, IPC_RMID, NULL);
      shmids_.erase(shmid);
    } else {
      printf("There is no shmem whose id is %i.\n", shmid);
    }
    return result;
  }

  int DeleteSharedMemory(const std::string &src_name) {
    // Release specific shared memory
    int result = -1;
    if (shms_.find(src_name) != shms_.end()) {
      result = shm_unlink(src_name.c_str());
      shms_.erase(src_name);
    } else {
      printf("There is no shmem named %s.\n", src_name);
    }
    return result;
  }

  int Stop() {
    int result = 0;
    for (auto it = shms_.begin(); it != shms_.end(); it++) {
      result = shm_unlink(it->c_str());
      if (result)
        exit(-1);
    }
    shms_.clear();
    for (auto it = shmids_.begin(); it != shmids_.end(); it++) {
      result = shmctl(*it, IPC_RMID, NULL);
      if (result)
        exit(-1);
    }
    shmids_.clear();
    return result;
  }

 private:
  std::set<std::string> shms_;
  std::set<int> shmids_;
  key_t key_ = 0xdead;
};

}  // namespace message_infrastructure

#endif  // SHM_H_
