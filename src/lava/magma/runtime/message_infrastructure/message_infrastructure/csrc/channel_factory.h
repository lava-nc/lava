// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_FACTORY_H_
#define CHANNEL_FACTORY_H_

#include <string>
#include <memory>

#include "abstract_channel.h"
#include "shmem_channel.h"
#include "utils.h"

namespace message_infrastructure {

class ChannelFactory {
 public:
  template<class T>
  std::shared_ptr<AbstractChannel> GetChannel(
      const ChannelType &channel_type,
      SharedMemoryPtr shm,
      const pybind11::array_t<T> &data,
      const size_t &size,
      const size_t &nbytes,
      const std::string &name = "test_channel") {
    switch (channel_type) {
      case RPCCHANNEL:
        break;
      case DDSCHANNEL:
        break;
      default:
        return GetShmemChannel<T>(shm, data, size, nbytes, name);
        break;
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
