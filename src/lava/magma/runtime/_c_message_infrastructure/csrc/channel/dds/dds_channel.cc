// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <channel/dds/dds_channel.h>
#include <channel/dds/dds.h>
#include <core/utils.h>
#include <core/message_infrastructure_logging.h>

namespace message_infrastructure {

DDSChannel::DDSChannel(const std::string &src_name,
                       const std::string &dst_name,
                       const size_t &size,
                       const size_t &nbytes,
                       const DDSTransportType &dds_transfer_type,
                       const DDSBackendType &dds_backend) {
  LAVA_DEBUG(LOG_DDS, "Creating DDSChannel...\n");

  dds_ = GetDDSManagerSingleton().AllocDDS(
                                  "dds_topic_" + std::to_string(std::rand()),
                                  dds_transfer_type,
                                  dds_backend,
                                  size);
  send_port_ = std::make_shared<DDSSendPort>(src_name, size, nbytes, dds_);
  recv_port_ = std::make_shared<DDSRecvPort>(dst_name, size, nbytes, dds_);
}

AbstractSendPortPtr DDSChannel::GetSendPort() {
  return send_port_;
}

AbstractRecvPortPtr DDSChannel::GetRecvPort() {
  return recv_port_;
}

std::shared_ptr<DDSChannel> GetDefaultDDSChannel(const size_t &nbytes,
                                                 const size_t &size,
                                                 const std::string &src_name,
                                                 const std::string &dst_name) {
  DDSBackendType BackendType = DDSBackendType::FASTDDSBackend;
  #if defined(CycloneDDS_ENABLE)
    BackendType = DDSBackendType::CycloneDDSBackend;
  #endif
  LAVA_LOG_ERR("GetDefaultDDSChannel function====\n");
  return std::make_shared<DDSChannel>(src_name,
                                      dst_name,
                                      size,
                                      nbytes,
                                      DDSTransportType::DDSUDPv4,
                                      BackendType);
}

}  // namespace message_infrastructure
