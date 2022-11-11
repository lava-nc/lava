// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_UTILS_H_
#define CORE_UTILS_H_

#include <memory>
#include <chrono>  // NOLINT
#include <thread>  // NOLINT

#if defined(ENABLE_MM_PAUSE)
#include <immintrin.h>
#endif

#define MAX_ARRAY_DIMS (5)
#define SLEEP_NS (1)

#define NPY_ATTR_DEPRECATE(text)

#define GET_METADATA(metadataptr, array, nd, dtype, dims) do { \
    if (nd > 0) { \
        for (int i = 0; i < nd ; i++) { \
            metadataptr->dims[i] = dims[i]; \
        } \
        int product = 1; \
        for (int i = 0; i < nd; i++) { \
            metadataptr->strides[nd-i-1] = product; \
            product*= metadataptr->dims[nd-i-1]; \
        } \
        metadataptr->total_size = product; \
        metadataptr->elsize = sizeof(*array); \
        metadataptr->type = dtype; \
        metadataptr->mdata = reinterpret_cast<void*>(array); \
    } \
} while (0)

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

enum NPY_TYPES {    NPY_BOOL = 0,
                    NPY_BYTE, NPY_UBYTE,
                    NPY_SHORT, NPY_USHORT,
                    NPY_INT, NPY_UINT,
                    NPY_LONG, NPY_ULONG,
                    NPY_LONGLONG, NPY_ULONGLONG,
                    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
                    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
                    NPY_OBJECT = 17,
                    NPY_STRING, NPY_UNICODE,
                    NPY_VOID,
                    /*
                     * New 1.6 types appended, may be integrated
                     * into the above in 2.0.
                     */
                    NPY_DATETIME, NPY_TIMEDELTA, NPY_HALF,

                    NPY_NTYPES,
                    NPY_NOTYPE,
                    NPY_CHAR NPY_ATTR_DEPRECATE("Use NPY_STRING"),
                    NPY_USERDEF = 256,  /* leave room for characters */

                    /* The number of types not including the new 1.6 types */
                    NPY_NTYPES_ABI_COMPATIBLE = 21
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
