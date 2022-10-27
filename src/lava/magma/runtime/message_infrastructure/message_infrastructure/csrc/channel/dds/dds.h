// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_DDS_H_
#define CHANNEL_DDS_DDS_H_

#include <message_infrastructure/csrc/core/utils.h>
#include <memory>
#include <set>

namespace message_infrastructure {
class DDSPublisher {
 public:
  ~DDSPublisher();
  virtual int Init() = 0;
  virtual bool Publish() = 0;
};

using DDSPublisherPtr = std::shared_ptr<DDSPublisher>;

class DDSSubscriber {
 public:
  ~DDSSubscriber();
  virtual int init() = 0;
  virtual MetaDataPtr Read() = 0;
};

using DDSSubscriberPtr = std::shared_ptr<DDSSubscriber>;

class DDS {
 public:
  DDS(const std::string &name, const int &depth, const size_t &mem_size);
  ~DDS();
 private:
  DDSPublisherPtr dds_publisher_ = nullptr;
  DDSSubscriberPtr dds_subscriber_ = nullptr;
};

using DDSPtr = std::shared_ptr<DDS>;

class DDSManager {
 public:
  ~DDSManager();
  std::shared_ptr<DDS> AllocDDS(const size_t &size);
  void DeleteDDS();
  friend DDSManager &GetDDSManager();

 private:
  DDSManager();
  std::set<std::string> ddss_;
};

DDSManager& GetDDSManager();
using DDSManagerPtr = std::shared_ptr<DDSManager>;

} // namespace message_infrastructure

#endif  // CHANNEL_DDS_DDS_H_