// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#if defined(FASTDDS_ENABLE)
#include <channel/dds/fast_dds.h>
#endif
#if defined(CycloneDDS_ENABLE)
#include <channel/dds/cyclone_dds.h>
#endif
#include <channel/dds/dds.h>
#include <core/utils.h>
#include <string>
#include <mutex>  // NOLINT

namespace message_infrastructure {
DDSPtr DDSManager::AllocDDS(const std::string &topic_name,
                            const DDSTransportType &dds_transfer_type,
                            const DDSBackendType &dds_backend,
                            const size_t &max_samples) {
  std::lock_guard<std::mutex> lg(dds_lock_);

  if (dds_topics_.find(topic_name) != dds_topics_.end()) {
    LAVA_LOG_ERR("The topic %s has already been used\n", topic_name.c_str());
    return nullptr;
  }
  dds_topics_.insert(topic_name);
  DDSPtr dds = std::make_shared<DDS>(topic_name,
                                     dds_transfer_type,
                                     dds_backend,
                                     max_samples);
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

void DDS::CreateFastDDSBackend(const std::string &topic_name,
                               const DDSTransportType &dds_transfer_type,
                               const size_t &max_samples) {
#if defined(FASTDDS_ENABLE)
  LAVA_LOG_ERR("DDS::CreateFastDDSBackend\n");
  dds_publisher_ = std::make_shared<FastDDSPublisher>(topic_name,
                                                      dds_transfer_type,
                                                      max_samples);
  dds_subscriber_ = std::make_shared<FastDDSSubscriber>(topic_name,
                                                        dds_transfer_type,
                                                        max_samples);
#else
  LAVA_LOG_FATAL("FastDDS is not enable, exit!\n");
#endif
}

void DDS::CreateCycloneDDSBackend(const std::string &topic_name,
                                  const DDSTransportType &dds_transfer_type,
                                  const size_t &max_samples) {
#if defined(CycloneDDS_ENABLE)
  dds_publisher_ = std::make_shared<CycloneDDSPublisher>(topic_name,
                                                         dds_transfer_type,
                                                         max_samples);
  dds_subscriber_ = std::make_shared<CycloneDDSSubscriber>(topic_name,
                                                           dds_transfer_type,
                                                           max_samples);
#else
  LAVA_LOG_FATAL("CycloneDDS is not enable, exit!\n");
#endif
}

DDS::DDS(const std::string &topic_name,
         const DDSTransportType &dds_transfer_type,
         const DDSBackendType &dds_backend,
         const size_t &max_samples) {
  if (dds_backend == DDSBackendType::FASTDDSBackend) {
    CreateFastDDSBackend(topic_name, dds_transfer_type, max_samples);
  } else if (dds_backend == DDSBackendType::CycloneDDSBackend) {
    CreateCycloneDDSBackend(topic_name, dds_transfer_type, max_samples);
  } else {
    LAVA_LOG_ERR("Not support DDSBackendType provided, %d\n",
                 static_cast<int>(dds_backend));
  }
}

DDSManager DDSManager::dds_manager_;

DDSManager& GetDDSManagerSingleton() {
  DDSManager &dds_manager = DDSManager::dds_manager_;
  return dds_manager;
}

}  // namespace message_infrastructure
