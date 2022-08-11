// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef INCLUDE_SHM_H_
#define INCLUDE_SHM_H_

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

}  // namespace message_infrastructure

#endif  // INCLUDE_SHM_H_
