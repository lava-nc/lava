// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_UTILS_H_
#define CORE_UTILS_H_

#include <condition_variable>  // NOLINT
#if defined(GRPC_CHANNEL)
#include <channel/grpc/grpcchannel.grpc.pb.h>
#endif
#include <core/message_infrastructure_logging.h>
#include <memory>
#include <chrono>  // NOLINT
#include <thread>  // NOLINT

#if defined(ENABLE_MM_PAUSE)
#include <immintrin.h>
#endif

#define MAX_ARRAY_DIMS (5)
#define SLEEP_NS (1)

#define LAVA_SIZEOF_CHAR       (sizeof(char))
#define LAVA_SIZEOF_UCHAR      (LAVA_SIZEOF_CHAR)

#define LAVA_SIZEOF_BOOL       (LAVA_SIZEOF_UCHAR)
#define LAVA_SIZEOF_BYTE       (LAVA_SIZEOF_CHAR)
#define LAVA_SIZEOF_UBYTE      (LAVA_SIZEOF_UCHAR)
#define LAVA_SIZEOF_SHORT      (sizeof(short))  // NOLINT
#define LAVA_SIZEOF_USHORT     (sizeof(u_short))
#define LAVA_SIZEOF_INT        (sizeof(int))
#define LAVA_SIZEOF_UINT       (sizeof(uint))
#define LAVA_SIZEOF_LONG       (sizeof(long))  // NOLINT
#define LAVA_SIZEOF_ULONG      (sizeof(ulong))
#define LAVA_SIZEOF_LONGLONG   (sizeof(long long))  // NOLINT
#define LAVA_SIZEOF_ULONGLONG  (sizeof(ulong long))  // NOLINT
#define LAVA_SIZEOF_FLOAT      (sizeof(float))
#define LAVA_SIZEOF_DOUBLE     (sizeof(double))
#define LAVA_SIZEOF_LONGDOUBLE (sizeof(long double))
#define LAVA_SIZEOF_NULL       (0)
// the length of string is unknown
#define LAVA_SIZEOF_STRING     (-1)

#define SIZEOF(TYPE) (LAVA_SIZEOF_ARRAY[TYPE])

namespace message_infrastructure {

enum class ProcessType {
  ErrorProcess = 0,
  ParentProcess = 1,
  ChildProcess = 2
};

enum class ChannelType {
  SHMEMCHANNEL = 0,
  RPCCHANNEL = 1,
  DDSCHANNEL = 2,
  SOCKETCHANNEL = 3,
  TEMPCHANNEL = 4
};

enum class METADATA_TYPES : int64_t{ BOOL = 0,
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

static int64_t LAVA_SIZEOF_ARRAY[static_cast<int>(METADATA_TYPES::STRING) + 1] =
  { LAVA_SIZEOF_BOOL,
    LAVA_SIZEOF_BYTE,
    LAVA_SIZEOF_UBYTE,
    LAVA_SIZEOF_SHORT,
    LAVA_SIZEOF_USHORT,
    LAVA_SIZEOF_INT,
    LAVA_SIZEOF_UINT,
    LAVA_SIZEOF_LONG,
    LAVA_SIZEOF_ULONG,
    LAVA_SIZEOF_LONGLONG,
    LAVA_SIZEOF_ULONGLONG,
    LAVA_SIZEOF_FLOAT,
    LAVA_SIZEOF_DOUBLE,
    LAVA_SIZEOF_LONGDOUBLE,
    LAVA_SIZEOF_NULL,
    LAVA_SIZEOF_NULL,
    LAVA_SIZEOF_NULL,
    LAVA_SIZEOF_NULL,
    LAVA_SIZEOF_STRING
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
    LAVA_LOG_ERR("Invalid nd: %ld\n", nd);
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

class LavaCondition {
 private:
  std::mutex mtx;
  std::condition_variable cv;
 public:
  void Waitfunc() {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [] {return true;});
  }
  void Notifyfunc() {
    std::unique_lock<std::mutex> lk(mtx);
    cv.notify_all();
  }
};

#if defined(DDS_CHANNEL)
// Default Parameters
// Transport
#define SHM_SEGMENT_SIZE (2 * 1024 * 1024)
#define NON_BLOCKING_SEND (false)
#define UDP_OUT_PORT  (0)
#define TCP_PORT 46
#define TCPv4_IP ("0.0.0.0")
// QOS
#define HEARTBEAT_PERIOD_SECONDS (2)
#define HEARTBEAT_PERIOD_NANOSEC (200 * 1000 * 1000)
// Topic
#define DDS_DATATYPE_NAME "ddsmetadata::msg::dds_::DDSMetaData_"
// Using default_nbytes in DDSChannel
// When user only care about the topic_name and not the nbytes
#define DEFAULT_NBYTES 0

enum class DDSTransportType {
  DDSSHM = 0,
  DDSTCPv4 = 1,
  DDSTCPv6 = 2,
  DDSUDPv4 = 3,
  DDSUDPv6 = 4
};

enum class DDSBackendType {
  FASTDDSBackend = 0,
  CycloneDDSBackend = 1
};

enum class DDSInitErrorType {
  DDSNOERR = 0,
  DDSParticipantError = 1,
  DDSPublisherError = 2,
  DDSSubscriberError = 3,
  DDSTopicError = 4,
  DDSDataWriterError = 5,
  DDSDataReaderError = 6,
  DDSTypeParserError = 7
};

#endif

#if defined(GRPC_CHANNEL)
#define DEFAULT_GRPC_URL "0.0.0.0:"
#define DEFAULT_GRPC_PORT 8000
#endif

}  // namespace message_infrastructure

#endif  // CORE_UTILS_H_
