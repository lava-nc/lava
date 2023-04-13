// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_CHANNEL_FACTORY_H_
#define CORE_CHANNEL_FACTORY_H_

#include <core/abstract_channel.h>
#include <core/utils.h>

#include <string>
#include <memory>

namespace message_infrastructure {

class ChannelFactory {
 public:
  ChannelFactory(const ChannelFactory&) = delete;
  ChannelFactory(ChannelFactory&&) = delete;
  ChannelFactory& operator=(const ChannelFactory&) = delete;
  ChannelFactory& operator=(ChannelFactory&&) = delete;

  AbstractChannelPtr GetChannel(const ChannelType &channel_type,
                                const size_t &size,
                                const size_t &nbytes,
                                const std::string &src_name,
                                const std::string &dst_name);

  AbstractChannelPtr GetTempChannel(const std::string &addr_path);
#if defined(DDS_CHANNEL)
  AbstractChannelPtr GetDDSChannel(const std::string &src_name,
                                   const std::string &dst_name,
                                   const size_t &size,
                                   const size_t &nbytes,
                                   const DDSTransportType &dds_transfer_type,
                                   const DDSBackendType &dds_backend);
#endif

#if defined(GRPC_CHANNEL)
  AbstractChannelPtr GetRPCChannel(const std::string &url,
                                   const int &port,
                                   const std::string &src_name,
                                   const std::string &dst_name,
                                   const size_t &size);

  AbstractChannelPtr GetDefRPCChannel(const std::string &src_name,
                                      const std::string &dst_name,
                                      const size_t &size);
#endif

  friend ChannelFactory& GetChannelFactory();

 private:
  ~ChannelFactory() = default;
  ChannelFactory() = default;
  static ChannelFactory channel_factory_;
};

ChannelFactory& GetChannelFactory();

}  // namespace message_infrastructure

#endif  // CORE_CHANNEL_FACTORY_H_
