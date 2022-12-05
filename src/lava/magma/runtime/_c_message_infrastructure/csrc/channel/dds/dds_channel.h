// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_DDS_CHANNEL_H_
#define CHANNEL_DDS_DDS_CHANNEL_H_

#include <core/abstract_channel.h>
#include <core/abstract_port.h>
#include <channel/dds/dds.h>
#include <channel/dds/dds_port.h>
#include <core/utils.h>

#include <memory>
#include <string>

namespace message_infrastructure {

class DDSChannel : public AbstractChannel {
 public:
  DDSChannel() = delete;
  ~DDSChannel() override {}
  DDSChannel(const std::string &topic_name,
            const DDSTransportType &dds_transfer_type,
            const DDSBackendType &dds_backend,
            const size_t &size);
  AbstractSendPortPtr GetSendPort();
  AbstractRecvPortPtr GetRecvPort();

 private:
  DDSPtr dds_ = nullptr;
  DDSSendPortPtr send_port_ = nullptr;
  DDSRecvPortPtr recv_port_ = nullptr;
};
}  // namespace message_infrastructure

#endif  // CHANNEL_DDS_DDS_CHANNEL_H_
