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

namespace message_infrastructure {

class SharedMemory {
};

using SharedMemoryPtr = SharedMemory *;

}  // namespace message_infrastructure

#endif  // SHM_H_
