// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "shmem_channel.h"

namespace message_infrastructure {

ShmemChannel::ShmemChannel(SharedMemoryPtr shm,
               const std::string &src_name,
               const std::string &dst_name,
               const ssize_t* shape,
               const pybind11::dtype &dtype,
               const size_t &size,
               const size_t &nbytes) {
  // Todo
}
}  // namespace message_infrastructure
