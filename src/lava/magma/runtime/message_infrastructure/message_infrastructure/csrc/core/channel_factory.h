// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_FACTORY_H_
#define CHANNEL_FACTORY_H_

#include <string>
#include <memory>

#include <message_infrastructure/csrc/core/abstract_channel.h>
#include <message_infrastructure/csrc/channel/shmem/shmem_channel.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/channel/shmem/shm.h>
#include <message_infrastructure/csrc/channel/socket/socket.h>
#include <message_infrastructure/csrc/channel/socket/socket_channel.h>


namespace message_infrastructure {

class ChannelFactory {
 public:
  AbstractChannelPtr GetChannel(
      const ChannelType &channel_type,
      const size_t &size,
      const size_t &nbytes,
      const std::string &src_name,
      const std::string &dst_name) {
    switch (channel_type) {
      case RPCCHANNEL:
        break;
      case DDSCHANNEL:
        break;
      case SOCKETCHANNEL:
        return GetSocketChannel(1, nbytes, src_name, dst_name);
      default:
        return GetShmemChannel(size, nbytes, src_name, dst_name);
    }
    return NULL;
  }
  friend ChannelFactory& GetChannelFactory();

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

#endif  // CHANNEL_FACTORY_H_
