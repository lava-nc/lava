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
  DDSChannel(const std::string &src_name,
            const std::string &dst_name,
            const size_t &size,
            const size_t &nbytes,
            const DDSTransportType &dds_transfer_type,
            const DDSBackendType &dds_backend);
  AbstractSendPortPtr GetSendPort();
  AbstractRecvPortPtr GetRecvPort();

 private:
  DDSPtr dds_ = nullptr;
  DDSSendPortPtr send_port_ = nullptr;
  DDSRecvPortPtr recv_port_ = nullptr;
};

std::shared_ptr<DDSChannel> GetDefaultDDSChannel(const size_t &nbytes,
                                                 const size_t &size,
                                                 const std::string &src_name,
                                                 const std::string &dst_name);

}  // namespace message_infrastructure

#endif  // CHANNEL_DDS_DDS_CHANNEL_H_
