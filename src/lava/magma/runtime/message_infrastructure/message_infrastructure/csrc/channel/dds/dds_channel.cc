// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/dds/dds_channel.h>
#include <message_infrastructure/csrc/channel/dds/dds.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <iostream>
namespace message_infrastructure {

DDSChannel::DDSChannel(const size_t &size,
                       const size_t &nbytes,
                       const std::string &topic_name) {
  size_t sample_bytes = nbytes + sizeof(MetaData);
  printf("DDSChannel: get dds\n");
  dds_ = GetDDSManager().AllocDDS(size, sample_bytes, topic_name);
  printf("DDSChannel: set port\n");
  send_port_ = std::make_shared<DDSSendPort>(dds_);
  recv_port_ = std::make_shared<DDSRecvPort>(dds_);
}

AbstractSendPortPtr DDSChannel::GetSendPort() {
  return send_port_;
}

AbstractRecvPortPtr DDSChannel::GetRecvPort() {
  return recv_port_;
}

std::shared_ptr<DDSChannel> GetDDSChannel(const size_t &size,
                              const size_t &nbytes,
                              const std::string &topic_name) {
  return (std::make_shared<DDSChannel>(size, nbytes, topic_name));
};
}
