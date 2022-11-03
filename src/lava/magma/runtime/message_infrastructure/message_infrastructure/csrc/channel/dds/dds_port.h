// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_DDS_DDS_PORT_H_
#define CHANNEL_DDS_DDS_PORT_H_

#include <message_infrastructure/csrc/core/abstract_port.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/channel/dds/dds.h>
#include <atomic>
#include <memory>

namespace message_infrastructure {

class DDSSendPort final : public AbstractSendPort {
 public:
  DDSSendPort(DDSPtr dds) : publisher_(dds->dds_publisher_) {}
  void Start() {
    int flag = publisher_->Init();
    if (flag) {
      LAVA_LOG_ERR("Publisher Init return error, %d\n", flag);
      exit(-1);
    }
    done_.store(false);
  }
  void Send(MetaDataPtr metadata) {
    while (!publisher_->Publish(metadata)) {
      helper::Sleep();
    }
  }
  void Join() {
    publisher_->Stop();
    done_.store(true);
  }
  bool Probe() {
    return false;
  }

 private:
  DDSPublisherPtr publisher_;
  std::atomic_bool done_;
};

using DDSSendPortPtr = std::shared_ptr<DDSSendPort>;

class DDSRecvPort final : public AbstractRecvPort {
 public:
  DDSRecvPort(DDSPtr dds) : subscriber_(dds->dds_subscriber_) {}
  void Start() {
    int flag = subscriber_->Init();
    if (flag) {
      LAVA_LOG_ERR("Publisher Init return error, %d\n", flag);
      exit(-1);
    }
    done_.store(false);
  }
  MetaDataPtr Recv() {
    return subscriber_->Read();
  }
  void Join() {
    subscriber_->Stop();
    done_.store(true);
  }
  MetaDataPtr Peek() {
    // RecvQueue not achieved, cannot just peek the data
    return Recv();
  }
  bool Probe() {
    return false;
  }

 private:
  DDSSubscriberPtr subscriber_;
  std::atomic_bool done_;
};

using DDSRecvPortPtr = std::shared_ptr<DDSRecvPort>;

}  // namespace message_infrastructure

#endif  // CHANNEL_DDS_DDS_PORT_H_
