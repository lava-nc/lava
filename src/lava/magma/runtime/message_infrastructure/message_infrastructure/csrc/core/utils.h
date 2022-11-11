// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_UTILS_H_
#define CORE_UTILS_H_

#include <message_infrastructure/csrc/channel/grpc/grpcchannel.grpc.pb.h>

#include <memory>
#include <chrono>  // NOLINT
#include <thread>  // NOLINT

#if defined(ENABLE_MM_PAUSE)
#include <immintrin.h>
#endif

#define MAX_ARRAY_DIMS (5)
#define SLEEP_NS (1)

namespace message_infrastructure {

enum ProcessType {
  ErrorProcess = 0,
  ParentProcess = 1,
  ChildProcess = 2
};

enum ChannelType {
  SHMEMCHANNEL = 0,
  RPCCHANNEL = 1,
  DDSCHANNEL = 2,
  SOCKETCHANNEL = 3
};

struct MetaData {
  int64_t nd;
  int64_t type;
  int64_t elsize;
  int64_t total_size;
  int64_t dims[MAX_ARRAY_DIMS] = {0};
  int64_t strides[MAX_ARRAY_DIMS] = {0};
  void* mdata;
};

// Incase Peek() and Recv() operations of ports will reuse Metadata.
// Use std::shared_ptr.
using MetaDataPtr = std::shared_ptr<MetaData>;
using grpcchannel::GrpcMetaData;
using GrpcMetaDataPtr = std::shared_ptr<GrpcMetaData>;
namespace helper {

static void Sleep() {
#if defined(ENABLE_MM_PAUSE)
  _mm_pause();
#else
  std::this_thread::sleep_for(std::chrono::nanoseconds(SLEEP_NS));
#endif
}
}
}  // namespace message_infrastructure

#endif  // CORE_UTILS_H_
