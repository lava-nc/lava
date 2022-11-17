// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_DDS_PORT_H_
#define CHANNEL_DDS_DDS_PORT_H_

#include <atomic>
#include <memory>

namespace message_infrastructure {

class DDSSendPort final : public AbstractSendPort {
 public:
  DDSSendPort(DDSPtr dds) : publisher_(dds->dds_publisher_) {}
  void Start() {
    int flag = publisher_->Init();
    if (flag) {
      LAVA_LOG_FATAL("Publisher Init return error, %d\n", flag);
    }
  }
  void Send(DataPtr data) {
    while (!publisher_->Publish(data)) {
      helper::Sleep();
    }
  }
  void Join() {
    publisher_->Stop();
  }
  bool Probe() {
    return false;
  }

 private:
  DDSPublisherPtr publisher_;
};

// Users should be allowed to copy port objects.
// Use std::shared_ptr.
using DDSSendPortPtr = std::shared_ptr<DDSSendPort>;

class DDSRecvPort final : public AbstractRecvPort {
 public:
  DDSRecvPort(DDSPtr dds) : subscriber_(dds->dds_subscriber_) {}
  void Start() {
    int flag = subscriber_->Init();
    if (flag) {
      LAVA_LOG_FATAL("Subscriber Init return error, %d\n", flag);
    }
  }
  MetaDataPtr Recv() {
    return subscriber_->Recv(false);
  }
  void Join() {
    subscriber_->Stop();
  }
  MetaDataPtr Peek() {
    return subscriber_->Recv(true);
  }
  bool Probe() {
    return false;
  }

 private:
  DDSSubscriberPtr subscriber_;
};

using DDSRecvPortPtr = std::shared_ptr<DDSRecvPort>;

}  // namespace message_infrastructure

#endif  // CHANNEL_DDS_DDS_PORT_H_
