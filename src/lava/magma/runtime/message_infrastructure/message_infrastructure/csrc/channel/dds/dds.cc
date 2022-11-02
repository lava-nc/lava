// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#if defined(FASTDDS)
#include <message_infrastructure/csrc/channel/dds/fast_dds.h>
#endif
#include <message_infrastructure/csrc/channel/dds/dds.h>
#include <string>

namespace message_infrastructure {
DDSPtr DDSManager::AllocDDS(const size_t &depth,
                     const size_t &nbytes,
                     const std::string &topic_name,
                     const DDSTransportType &dds_transfer_type,
                     const DDSBackendType &dds_backend) {
  if (dds_topics_.find(topic_name) != dds_topics_.end()) {
    LAVA_LOG_ERR("The topic %s has already been used\n", topic_name.c_str());
    return nullptr;
  }
  dds_topics_.insert(topic_name);
  DDSPtr dds = std::make_shared<DDS>(depth,
                                     nbytes,
                                     topic_name,
                                     dds_transfer_type,
                                     dds_backend);
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

void DDS::CreateFastDDSBackend(const size_t &max_samples,
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
#else
  LAVA_LOG_ERR("CycloneDDS is not enable, exit!\n");
  exit(-1);
#endif
}

void DDS::CreateCycloneDDSBackend(const size_t &max_samples,
                                  const size_t &nbytes,
                                  const std::string &topic_name,
                                  const DDSTransportType &dds_transfer_type) {
  LAVA_LOG_ERR("CycloneDDS is not enable, exit!\n");
  exit(-1);
}

DDS::DDS(const size_t &max_samples,
         const size_t &nbytes,
         const std::string &topic_name,
         const DDSTransportType &dds_transfer_type,
         const DDSBackendType &dds_backend) {
  if (dds_backend == FASTDDSBackend) {
    CreateFastDDSBackend(max_samples, nbytes, topic_name, dds_transfer_type);
  } else if (dds_backend == CycloneDDSBackend) {
    CreateCycloneDDSBackend(max_samples, nbytes, topic_name, dds_transfer_type);
  } else {
    LAVA_LOG_ERR("Not support DDSBackendType provided, %d\n", dds_backend);
  }
}

DDSManager DDSManager::dds_manager_;

DDSManager& GetDDSManager() {
  DDSManager &dds_manager = DDSManager::dds_manager_;
  return dds_manager;
}

}  // namespace message_infrastructure
