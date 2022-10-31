// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_DDS_CHANNEL_H_
#define CHANNEL_DDS_DDS_CHANNEL_H_

#include <message_infrastructure/csrc/core/abstract_channel.h>
#include <message_infrastructure/csrc/core/abstract_port.h>
#include <message_infrastructure/csrc/channel/dds/dds.h>
#include <message_infrastructure/csrc/channel/dds/dds_port.h>
#include <message_infrastructure/csrc/core/utils.h>

#include <memory>
#include <string>

namespace message_infrastructure {

class DDSChannel : public AbstractChannel {
 public:
  DDSChannel(){}
  DDSChannel(const size_t &depth,
             const size_t &nbytes,
             const std::string &topic_name,
             const DDSTransportType &dds_transfer_type);
  AbstractSendPortPtr GetSendPort();
  AbstractRecvPortPtr GetRecvPort();

 private:
  DDSPtr dds_ = nullptr;
  DDSSendPortPtr send_port_ = nullptr;
  DDSRecvPortPtr recv_port_ = nullptr;
};
}  // namespace message_infrastructure

#endif  // CHANNEL_DDS_DDS_CHANNEL_H_