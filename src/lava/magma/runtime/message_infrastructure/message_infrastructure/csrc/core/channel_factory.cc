// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/core/channel_factory.h>

namespace message_infrastructure {

AbstractChannelPtr ChannelFactory::GetChannel(const ChannelType &channel_type,
                                              const size_t &size,
                                              const size_t &nbytes,
                                              const std::string &src_name,
                                              const std::string &dst_name) {
  switch (channel_type) {
    case DDSCHANNEL:
      break;
    case SOCKETCHANNEL:
      return GetSocketChannel(nbytes, src_name, dst_name);
    default:
      return GetShmemChannel(size, nbytes, src_name, dst_name);
  }
  return NULL;
}

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

extern ChannelFactory ChannelFactory::channel_factory_;

ChannelFactory& GetChannelFactory() {
  ChannelFactory &channel_factory = ChannelFactory::channel_factory_;
  return channel_factory;
}

}  // namespace message_infrastructure
