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

bool SocketWrite(int fd, void* data, size_t size) {
  size_t length = write(fd, (char *)data, size);

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

bool SocketRead(int fd, void* data, size_t size) {
  size_t length = read(fd, (char *)data, size);

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
  bool ret = false;
  while (!ret) {
    ret = SocketWrite(socket_.first, metadata.get(), sizeof(MetaData));
  }
  ret = false;
  while(!ret) {
    ret = SocketWrite(socket_.first, metadata->mdata, nbytes_);
  }
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
  bool ret = false;
  MetaDataPtr metadata = std::make_shared<MetaData>();
  ret = SocketRead(socket_.second, metadata.get(), sizeof(MetaData));
  if (!ret){
    metadata.reset();
    return metadata;
  }
  void *mdata = malloc(nbytes_);
  ret = SocketRead(socket_.second, mdata, nbytes_);
  metadata->mdata = mdata;
  if(!ret) {
    metadata.reset();
    free(mdata);
  }
  return metadata;
}
void SocketRecvPort::Join() {
  close(socket_.first);
  close(socket_.second);
}
MetaDataPtr SocketRecvPort::Peek() {
  return Recv();
}

}  // namespace message_infrastructure
