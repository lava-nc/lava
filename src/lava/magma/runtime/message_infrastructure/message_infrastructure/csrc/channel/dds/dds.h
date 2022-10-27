// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_DDS_H_
#define CHANNEL_DDS_DDS_H_

#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <memory>
#include <set>

namespace message_infrastructure {
class DDSPublisher {
 public:
  virtual int Init() = 0;
  virtual bool Publish(MetaDataPtr metadata) = 0;
};

using DDSPublisherPtr = std::shared_ptr<DDSPublisher>;

class DDSSubscriber {
 public:
  virtual int Init() = 0;
  virtual MetaDataPtr Read() = 0;
};

using DDSSubscriberPtr = std::shared_ptr<DDSSubscriber>;

class DDS {
 public:
  DDS(const size_t &max_samples, const size_t &nbytes, const std::string &topic_name);
  DDSPublisherPtr dds_publisher_ = nullptr;
  DDSSubscriberPtr dds_subscriber_ = nullptr;
};

using DDSPtr = std::shared_ptr<DDS>;

class DDSManager {
 public:
  ~DDSManager();
  DDSPtr AllocDDS(const size_t &size,
                       const size_t &nbytes,
                       const std::string &topic_name);
  void DeleteAllDDS();
  friend DDSManager &GetDDSManager();

 private:
  DDSManager(){};
  std::vector<DDSPtr> ddss_;
  std::set<std::string> dds_topics_;
  static DDSManager dds_manager_;
};

DDSManager& GetDDSManager();
using DDSManagerPtr = std::shared_ptr<DDSManager>;

} // namespace message_infrastructure

#endif  // CHANNEL_DDS_DDS_H_