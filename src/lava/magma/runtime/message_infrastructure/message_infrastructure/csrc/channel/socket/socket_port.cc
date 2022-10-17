// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <semaphore.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <memory>
#include <string>
#include <condition_variable>
#include <cassert>
#include <cstring>

#include <message_infrastructure/csrc/channel/socket/socket_port.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

namespace message_infrastructure {

bool SocketWriteValid(size_t length, size_t size) {
  if (length == -1) {
    LAVA_LOG_ERR("Write socket failed.\n");
    return false;
  }
  else if (length != size) {
    LAVA_LOG_ERR("Write socket error, expected size: %zd, got size: %zd", size, length);
    return false;
  }
  return true;
}

bool SocketReadValid(size_t length, size_t size) {
  if (length == -1) {
    LAVA_LOG_ERR("Read socket failed.\n");
    return false;
  }
  else if (length != size) {
    LAVA_LOG_ERR("Read socket error, expected size: %zd, got size: %zd", size, length);
    return false;
  }
  return true;
}

void SocketSendPort::Start() {}
void SocketSendPort::Send(MetaDataPtr metadata) {
  size_t length = write(socket_.first, (char*)metadata.get(), sizeof(MetaData));
  SocketWriteValid(length, sizeof(MetaData));
  length = write(socket_.first, metadata->mdata, nbytes_);
  SocketWriteValid(length, nbytes_);
}
void SocketSendPort::Join() {
  close(socket_.first);
  close(socket_.second);
}
bool SocketSendPort::Probe() {
  return false;
}

void SocketRecvPort::Start() {}
bool SocketRecvPort::Probe() {
  return false;
}
MetaDataPtr SocketRecvPort::Recv() {
  MetaDataPtr metadata = std::make_shared<MetaData>();
  size_t length = read(socket_.second, metadata.get(), sizeof(MetaData));
  SocketReadValid(length, sizeof(MetaData));
  void *mdata = malloc(nbytes_);
  length = read(socket_.second, mdata, nbytes_);
  metadata->mdata = mdata;
  SocketReadValid(length, nbytes_);
  return metadata;
}
void SocketRecvPort::Join() {
  close(socket_.first);
  close(socket_.second);
}
MetaDataPtr SocketRecvPort::Peek() {
  MetaDataPtr metadata = std::make_shared<MetaData>();
  size_t length = read(socket_.second, metadata.get(), sizeof(MetaData));
  SocketReadValid(length, sizeof(MetaData));
  void *mdata = malloc(nbytes_);
  length = read(socket_.second, mdata, nbytes_);
  metadata->mdata = mdata;
  SocketReadValid(length, nbytes_);
  return metadata;
}
}  // namespace message_infrastructure
