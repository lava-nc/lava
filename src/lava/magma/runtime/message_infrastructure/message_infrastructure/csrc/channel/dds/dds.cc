// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifdef fast_dds
#include <message_infrastructure/csrc/channel/dds/fast_dds.h>
#endif
#include <message_infrastructure/csrc/channel/dds/dds.h>

namespace message_infrastructure {
DDSPtr DDSManager::AllocDDS(const size_t &size,
                     const size_t &nbytes,
                     const std::string &topic_name) {
  if (dds_topics_.find(topic_name) != dds_topics_.end()) {
    LAVA_LOG_ERR("The topic %s has already been used\n", topic_name.c_str());
    return nullptr;
  }
  dds_topics_.insert(topic_name);
  DDSPtr dds = std::make_shared<DDS>(size, nbytes, topic_name);
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

DDS::DDS(const size_t &max_samples, const size_t &nbytes, const std::string &topic_name) {
#ifdef fast_dds
  dds_publisher_ = std::make_shared<FastDDSPublisher>(max_samples, nbytes, topic_name);
  dds_subscriber_ = std::make_shared<FastDDSSubscriber>(max_samples, nbytes, topic_name);
#endif
}


}  // message_infrastructure