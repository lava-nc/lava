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
#include <message_infrastructure/csrc/channel/grpc/grpc_channel.h>

#include <string>
#include <memory>

namespace message_infrastructure {

class ChannelFactory {
 public:
  AbstractChannelPtr GetChannel(const ChannelType &channel_type,
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

  friend ChannelFactory& GetChannelFactory();
#ifdef GRPC_CHANNEL
  AbstractChannelPtr GetRPCChannel(const std::string &url,
                                   const int &port,
                                   const std::string &src_name,
                                   const std::string &dst_name,
                                   const size_t &size) {
    return std::make_shared<GrpcChannel>(url, port, src_name, dst_name, size);
  }

  AbstractChannelPtr GetDefRPCChannel(const std::string &src_name,
                                      const std::string &dst_name,
                                      const size_t &size) {
    return std::make_shared<GrpcChannel>(src_name, dst_name, size);
  }
#endif

 private:
  ChannelFactory() {}
  ChannelFactory(const ChannelFactory&) {}
  static ChannelFactory channel_factory_;
};

ChannelFactory ChannelFactory::channel_factory_;

ChannelFactory& GetChannelFactory() {
  ChannelFactory &channel_factory = ChannelFactory::channel_factory_;
  return channel_factory;
}

}  // namespace message_infrastructure

#endif  // CORE_CHANNEL_FACTORY_H_
