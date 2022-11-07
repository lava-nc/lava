// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_CHANNEL_FACTORY_H_
#define CORE_CHANNEL_FACTORY_H_

#include <message_infrastructure/csrc/core/abstract_channel.h>
#include <message_infrastructure/csrc/channel/shmem/shmem_channel.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/channel/shmem/shm.h>
#include <message_infrastructure/csrc/channel/socket/socket.h>
#include <message_infrastructure/csrc/channel/socket/socket_channel.h>
#if defined(GRPC_CHANNEL)
#include <message_infrastructure/csrc/channel/grpc/grpc_channel.h>
#endif

#ifdef DDS_CHANNEL
#include <message_infrastructure/csrc/channel/dds/dds_channel.h>
#endif

#include <string>
#include <memory>

namespace message_infrastructure {

class ChannelFactory {
 public:
  AbstractChannelPtr GetChannel(const ChannelType &channel_type,
                                const size_t &size,
                                const size_t &nbytes,
                                const std::string &src_name,
                                const std::string &dst_name);

#if defined(DDS_CHANNEL)
  AbstractChannelPtr GetDDSChannel(const std::string &topic_name,
                                   const DDSTransportType &transport_type,
                                   const DDSBackendType &dds_backend,
                                   const size_t &size);
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
  ChannelFactory() {}
  ChannelFactory(const ChannelFactory&) {}
  static ChannelFactory channel_factory_;
};

ChannelFactory& GetChannelFactory();

}  // namespace message_infrastructure

#endif  // CORE_CHANNEL_FACTORY_H_
