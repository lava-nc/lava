// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_FACTORY_H_
#define CHANNEL_FACTORY_H_

#include <string>

#include "abstract_channel.h"
#include "shmem_channel.h"
#include "utils.h"

namespace message_infrastructure {

class ChannelFactory {
 public:
  ChannelFactory(const ChannelFactory&) = delete;
  ChannelFactory& operator=(const ChannelFactory&) = delete;
  static ChannelFactory& GetChannelFactory() {
    static ChannelFactory channel_factory;
    return channel_factory;
  }
  template<class T>
  AbstractChannelPtr GetChannel(
      const ChannelType &channel_type,
      SharedMemoryPtr shm,
      const pybind11::array_t<T> &data,
      const size_t &size,
      const std::string &name = "test_channel") {
    switch (channel_type) {
      case RPCCHANNEL:
        break;
      case DDSCHANNEL:
        break;
      default:
        return GetShmemChannel<T>(shm, data, size, name);
    }
    return NULL;
  }

 private:
  ChannelFactory();
};

}  // namespace message_infrastructure

#endif  // CHANNEL_FACTORY_H_
