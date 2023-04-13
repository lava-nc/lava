// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <core/channel_factory.h>
#include <channel/shmem/shm.h>
#include <channel/socket/socket.h>
#include <channel/socket/socket_channel.h>
#include <channel/shmem/shmem_channel.h>
#if defined(GRPC_CHANNEL)
#include <channel/grpc/grpc_channel.h>
#endif

#if defined(DDS_CHANNEL)
#include <channel/dds/dds_channel.h>
#endif
namespace message_infrastructure {

AbstractChannelPtr ChannelFactory::GetChannel(const ChannelType &channel_type,
                                              const size_t &size,
                                              const size_t &nbytes,
                                              const std::string &src_name,
                                              const std::string &dst_name) {
  switch (channel_type) {
#if defined(DDS_CHANNEL)
    case ChannelType::DDSCHANNEL:
      return GetDefaultDDSChannel(nbytes, size, src_name, dst_name);
#endif
    case ChannelType::SOCKETCHANNEL:
      return GetSocketChannel(nbytes, src_name, dst_name);
    default:
      return GetShmemChannel(size, nbytes, src_name, dst_name);
  }
  return nullptr;
}

AbstractChannelPtr ChannelFactory::GetTempChannel(
                                   const std::string &addr_path) {
  return std::make_shared<TempSocketChannel>(addr_path);
}

#if defined(DDS_CHANNEL)
AbstractChannelPtr ChannelFactory::GetDDSChannel(
                                  const std::string &src_name,
                                  const std::string &dst_name,
                                  const size_t &size,
                                  const size_t &nbytes,
                                  const DDSTransportType &dds_transfer_type,
                                  const DDSBackendType &dds_backend) {
  return std::make_shared<DDSChannel>(src_name,
                                      dst_name,
                                      size,
                                      nbytes,
                                      dds_transfer_type,
                                      dds_backend);
}
#endif

#if defined(GRPC_CHANNEL)
AbstractChannelPtr ChannelFactory::GetRPCChannel(const std::string &url,
                                                 const int &port,
                                                 const std::string &src_name,
                                                 const std::string &dst_name,
                                                 const size_t &size) {
  return std::make_shared<GrpcChannel>(url, port, src_name, dst_name, size);
}

AbstractChannelPtr ChannelFactory::GetDefRPCChannel(const std::string &src_name,
                                                    const std::string &dst_name,
                                                    const size_t &size) {
  return std::make_shared<GrpcChannel>(src_name, dst_name, size);
}
#endif

ChannelFactory ChannelFactory::channel_factory_;

ChannelFactory& GetChannelFactory() {
  ChannelFactory &channel_factory = ChannelFactory::channel_factory_;
  return channel_factory;
}

}  // namespace message_infrastructure
