// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_DDS_H_
#define CHANNEL_DDS_DDS_H_

#include <core/utils.h>
#include <core/message_infrastructure_logging.h>
#include <memory>
#include <unordered_set>
#include <string>
#include <vector>
#include <mutex>  // NOLINT

namespace message_infrastructure {
class DDSPublisher {
 public:
  virtual DDSInitErrorType Init() = 0;
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
  virtual DDSInitErrorType Init() = 0;
  virtual MetaDataPtr Recv(bool keep) = 0;
  virtual bool Probe() = 0;
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
  std::mutex dds_lock_;
  std::vector<DDSPtr> ddss_;
  std::unordered_set<std::string> dds_topics_;
  static DDSManager dds_manager_;
};

DDSManager& GetDDSManagerSingleton();

}  // namespace message_infrastructure

#endif  // CHANNEL_DDS_DDS_H_
