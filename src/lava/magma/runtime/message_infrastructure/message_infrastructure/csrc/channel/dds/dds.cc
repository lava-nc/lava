// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#if defined(FASTDDS)
#include <message_infrastructure/csrc/channel/dds/fast_dds.h>
#endif
#include <message_infrastructure/csrc/channel/dds/dds.h>

namespace message_infrastructure {
DDSPtr DDSManager::AllocDDS(const size_t &depth,
                     const size_t &nbytes,
                     const std::string &topic_name,
                     const DDSTransportType &dds_transfer_type) {
  if (dds_topics_.find(topic_name) != dds_topics_.end()) {
    LAVA_LOG_ERR("The topic %s has already been used\n", topic_name.c_str());
    return nullptr;
  }
  dds_topics_.insert(topic_name);
  DDSPtr dds = std::make_shared<DDS>(depth, nbytes, topic_name, dds_transfer_type);
  ddss_.push_back(dds);
  return dds;
}

void DDSManager::DeleteAllDDS() {
  ddss_.clear();
  dds_topics_.clear();
}

DDSManager::~DDSManager() {
  DeleteAllDDS();
}

DDS::DDS(const size_t &max_samples,
         const size_t &nbytes,
         const std::string &topic_name,
         const DDSTransportType &dds_transfer_type) {
#if defined(FASTDDS)
  dds_publisher_ = std::make_shared<FastDDSPublisher>(max_samples,
                                                      nbytes,
                                                      topic_name,
                                                      dds_transfer_type);
  dds_subscriber_ = std::make_shared<FastDDSSubscriber>(max_samples,
                                                        nbytes,
                                                        topic_name,
                                                        dds_transfer_type);
#endif
}

DDSManager DDSManager::dds_manager_;

DDSManager& GetDDSManager() {
  DDSManager &dds_manager = DDSManager::dds_manager_;
  return dds_manager;
}
}  // message_infrastructure