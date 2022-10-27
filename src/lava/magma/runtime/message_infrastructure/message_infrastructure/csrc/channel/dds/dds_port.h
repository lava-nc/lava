// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_DDS_PORT_H_
#define CHANNEL_DDS_DDS_PORT_H_

#include <message_infrastructure/csrc/core/abstract_port.h>
#include <atomic>

namespace message_infrastructure {

class DDSSendPort final : public AbstractSendPort {
 public:
  DDSSendPort(const std::string &name,
              const size_t &size,
              const size_t &nbytes);
  void Start();
  void Send();
  void Join();
  bool Probe();

 private:
  std::atomic_bool done_;
};

using DDSSendPortPtr = std::shared_ptr<DDSSendPort>;

class DDSRecvPort final : public AbstractRecvPort {
 public:
  DDSRecvPort(const std::string &name,
              const size_t &size,  // TODO: needn't any more
              const size_t &nbytes);
  void Start();
  bool Probe();
  MetaDataPtr Recv();
  void Join();
  MetaDataPtr Peek();
  void QueueRecv();  // TODO: use the common queue.

 private:
  std::atomic_bool done_;
};

using DDSRecvPortPtr = std::shared_ptr<DDSRecvPort>;

}  // namespace message_infrastructure

#endif  //CHANNEL_DDS_DDS_PORT_H_