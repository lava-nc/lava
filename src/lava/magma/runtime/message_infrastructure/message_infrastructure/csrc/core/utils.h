// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_UTILS_H_
#define CORE_UTILS_H_

#if defined(GRPC_CHANNEL)
#include <message_infrastructure/csrc/channel/grpc/grpcchannel.grpc.pb.h>
#endif
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <memory>
#include <chrono>  // NOLINT
#include <thread>  // NOLINT

#if defined(ENABLE_MM_PAUSE)
#include <immintrin.h>
#endif

#define MAX_ARRAY_DIMS (5)
#define SLEEP_NS (1)

#define SIZEOF_CHAR       (sizeof(char))
#define SIZEOF_UCHAR      (SIZEOF_CHAR)

#define SIZEOF_BOOL       (SIZEOF_UCHAR)
#define SIZEOF_BYTE       (SIZEOF_CHAR)
#define SIZEOF_UBYTE      (SIZEOF_UCHAR)
#define SIZEOF_SHORT      (sizeof(short))  // NOLINT
#define SIZEOF_USHORT     (sizeof(u_short))
#define SIZEOF_INT        (sizeof(int))
#define SIZEOF_UINT       (sizeof(uint))
#define SIZEOF_LONG       (sizeof(long))  // NOLINT
#define SIZEOF_ULONG      (sizeof(ulong))
#define SIZEOF_LONGLONG   (sizeof(long long))  // NOLINT
#define SIZEOF_ULONGLONG  (sizeof(ulong long))  // NOLINT
#define SIZEOF_FLOAT      (sizeof(float))
#define SIZEOF_DOUBLE     (sizeof(double))
#define SIZEOF_LONGDOUBLE (sizeof(long double))
#define SIZEOF_NULL       (0)
// the length of string is unknown
#define SIZEOF_STRING     (-1)

#define SIZEOF(TYPE) (SIZEOF_ARRAY[TYPE])

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

enum METADATA_TYPES { BOOL = 0,
                      BYTE, UBYTE,
                      SHORT, USHORT,
                      INT, UINT,
                      LONG, ULONG,
                      LONGLONG, ULONGLONG,
                      FLOAT, DOUBLE, LONGDOUBLE,
                      // align the value of STRING to
                      // NPY_STRING in ndarraytypes.h
                      STRING = 18
};

static int64_t SIZEOF_ARRAY[message_infrastructure::METADATA_TYPES::STRING+1] =
  { SIZEOF_BOOL,
    SIZEOF_BYTE,
    SIZEOF_UBYTE,
    SIZEOF_SHORT,
    SIZEOF_USHORT,
    SIZEOF_INT,
    SIZEOF_UINT,
    SIZEOF_LONG,
    SIZEOF_ULONG,
    SIZEOF_LONGLONG,
    SIZEOF_ULONGLONG,
    SIZEOF_FLOAT,
    SIZEOF_DOUBLE,
    SIZEOF_LONGDOUBLE,
    SIZEOF_NULL,
    SIZEOF_NULL,
    SIZEOF_NULL,
    SIZEOF_NULL,
    SIZEOF_STRING
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
using DataPtr = std::shared_ptr<void>;

#if defined(GRPC_CHANNEL)
using grpcchannel::GrpcMetaData;
using GrpcMetaDataPtr = std::shared_ptr<GrpcMetaData>;
#endif

inline void GetMetadata(const MetaDataPtr &metadataptr,
                        void *array,
                        const int64_t &nd,
                        const int64_t &dtype,
                        int64_t *dims) {
  if (nd <= 0 || nd > MAX_ARRAY_DIMS) {
    LAVA_LOG_ERR("invalid nd: %ld\n", nd);
    return;
  }
  for (int i = 0; i < nd ; i++) {
      metadataptr->dims[i] = dims[i];
  }
  int product = 1;
  for (int i = 0; i < nd; i++) {
      metadataptr->strides[nd - i - 1] = product;
      product *= metadataptr->dims[nd - i - 1];
  }
  metadataptr->total_size = product;
  metadataptr->elsize = SIZEOF(dtype);
  metadataptr->type = dtype;
  metadataptr->mdata = array;
}

inline void GetMetadata(const MetaDataPtr &metadataptr,
                        void *array,
                        const int64_t &nd,
                        const int64_t &dtype,
                        int64_t *dims) {
  if (nd <= 0 || nd > MAX_ARRAY_DIMS) {
    LAVA_LOG_ERR("invalid nd: %ld\n", nd);
    return;
  }
  for (int i = 0; i < nd ; i++) {
      metadataptr->dims[i] = dims[i];
  }
  int product = 1;
  for (int i = 0; i < nd; i++) {
      metadataptr->strides[nd - i - 1] = product;
      product *= metadataptr->dims[nd - i - 1];
  }
  metadataptr->total_size = product;
  metadataptr->elsize = SIZEOF(dtype);
  metadataptr->type = dtype;
  metadataptr->mdata = array;
}

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
