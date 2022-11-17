// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_DDS_H_
#define CHANNEL_DDS_DDS_H_

#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <memory>
#include <unordered_set>
#include <string>
#include <vector>

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

namespace message_infrastructure {

enum DDSTransportType {
  DDSSHM = 0,
  DDSTCPv4 = 1,
  DDSTCPv6 = 2,
  DDSUDPv4 = 3,
  DDSUDPv6 = 4
};

enum DDSBackendType {
  FASTDDSBackend = 0,
  CycloneDDSBackend = 1
};

enum DDSInitErrorType {
  DDSParticipantError = 1,
  DDSPublisherError = 2,
  DDSSubscriberError = 3,
  DDSTopicError = 4,
  DDSDataWriterError = 5,
  DDSDataReaderError = 6,
  DDSTypeParserError = 7
};

class DDSPublisher {
 public:
  virtual int Init() = 0;
  virtual bool Publish(DataPtr data) = 0;
  virtual void Stop() = 0;
  virtual ~DDSPublisher() {}
};

// DDSPublisher object needs to be transfered to DDSPort.
// Also need to be handled in DDS class.
// Use std::shared_ptr.
using DDSPublisherPtr = std::shared_ptr<DDSPublisher>;

class DDSSubscriber {
 public:
  virtual int Init() = 0;
  virtual MetaDataPtr Recv(bool keep) = 0;
  virtual void Stop() = 0;
  virtual ~DDSSubscriber() {}
};

// DDSSubscriber object needs to be transfered to DDSPort.
// Also need to be handled in DDS class.
// Use std::shared_ptr.
using DDSSubscriberPtr = std::shared_ptr<DDSSubscriber>;

class DDS {
 public:
  DDS(const std::string &topic_name,
      const DDSTransportType &dds_transfer_type,
      const DDSBackendType &dds_backend,
      const size_t &max_samples);
  DDSPublisherPtr dds_publisher_ = nullptr;
  DDSSubscriberPtr dds_subscriber_ = nullptr;

 private:
  void CreateFastDDSBackend(const std::string &topic_name,
                            const DDSTransportType &dds_transfer_type,
                            const size_t &max_samples);
  void CreateCycloneDDSBackend(const std::string &topic_name,
                               const DDSTransportType &dds_transfer_type,
                               const size_t &max_samples);
};

// DDS object needs to be transfered to DDSPort.
// Also need to be handled in DDSManager.
// Use std::shared_ptr.
using DDSPtr = std::shared_ptr<DDS>;

class DDSManager {
 public:
  DDSManager(const DDSManager&) = delete;
  DDSManager(DDSManager&&) = delete;
  DDSManager& operator=(const DDSManager&) = delete;
  DDSManager& operator=(DDSManager&&) = delete;
  DDSPtr AllocDDS(const std::string &topic_name,
                  const DDSTransportType &dds_transfer_type,
                  const DDSBackendType &dds_backend,
                  const size_t &max_samples);
  void DeleteAllDDS();
  friend DDSManager &GetDDSManagerSingleton();

 private:
  DDSManager() = default;
  ~DDSManager();
  std::vector<DDSPtr> ddss_;
  std::unordered_set<std::string> dds_topics_;
  static DDSManager dds_manager_;
};

DDSManager& GetDDSManagerSingleton();

// DDSManager object should be handled by multiple actors.
// Use std::shared_ptr.
using DDSManagerPtr = std::shared_ptr<DDSManager>;

}  // namespace message_infrastructure

#endif  // CHANNEL_DDS_DDS_H_
