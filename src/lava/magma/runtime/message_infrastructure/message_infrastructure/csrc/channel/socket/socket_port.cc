// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/socket/socket_port.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

#include <semaphore.h>
#include <sys/un.h>
#include <unistd.h>
#include <thread>  // NOLINT
#include <mutex>  // NOLINT
#include <memory>
#include <string>
#include <condition_variable>  // NOLINT
#include <cassert>
#include <cstring>

namespace message_infrastructure {

bool SocketWrite(int fd, void* data, size_t size) {
  size_t length = write(fd, reinterpret_cast<char *>(data), size);

  if (length != size) {
    if (length == -1) {
      LAVA_LOG_ERR("Write socket failed.\n");
      return false;
    }
    LAVA_LOG_ERR("Write socket error, expected size: %zd, got size: %zd",
                 size, length);
    return false;
  }
  return true;
}

bool SocketRead(int fd, void* data, size_t size) {
  size_t length = read(fd, reinterpret_cast<char *>(data), size);

  if (length != size) {
    if (length == -1) {
      LAVA_LOG_ERR("Read socket failed.\n");
      return false;
    }
    LAVA_LOG_ERR("Read socket error, expected size: %zd, got size: %zd",
                 size, length);
    return false;
  }
  return true;
}

void SocketSendPort::Start() {}
void SocketSendPort::Send(DataPtr metadata) {
  bool ret = false;
  while (!ret) {
    ret = SocketWrite(socket_.first,
                      reinterpret_cast<MetaData*>(metadata.get()),
                      sizeof(MetaData));
  }
  ret = false;
  while (!ret) {
    ret = SocketWrite(socket_.first,
                      reinterpret_cast<MetaData*>(metadata.get())->mdata,
                      nbytes_);
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
  if (!ret) {
    metadata.reset();
    return metadata;
  }
  void *mdata = std::calloc(nbytes_, 1);
  if (mdata == nullptr) {
    LAVA_LOG_FATAL("Memory alloc failed, errno: %d\n", errno);
  }
  ret = SocketRead(socket_.second, mdata, nbytes_);
  metadata->mdata = mdata;
  if (!ret) {
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

TempSocketSendPort::TempSocketSendPort(const SocketFile &addr_path) {
  name_ = "SendPort" + addr_path;
  addr_path_ = addr_path;
  cfd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  if (cfd_ == -1) {
    LAVA_LOG_ERR("Cannot Create Socket Domain File Descripter\n");
  }

  size_t skt_addr_len = sizeof(sa_family_t) + addr_path_.size();
  sockaddr *skt_addr = reinterpret_cast<sockaddr*>(malloc(skt_addr_len));
  skt_addr->sa_family = AF_UNIX;
  memcpy(skt_addr->sa_data, addr_path.c_str(), addr_path_.size());

  if (connect(cfd_, skt_addr, skt_addr_len) == -1) {
    LAVA_LOG_ERR("Cannot bind socket domain\n");
  }
}
void TempSocketSendPort::Start() {}
bool TempSocketSendPort::Probe() {
  LAVA_LOG_ERR("Not Support TempSocket Port Probe()\n");
  return false;
}
void TempSocketSendPort::Send(DataPtr data) {
  auto metadata = reinterpret_cast<MetaData*>(data.get());
  bool flag;
  flag = SocketWrite(cfd_, metadata, sizeof(MetaData));
  if (!flag) {
    LAVA_LOG_ERR("TempSkt Send data header Error\n");
  }
  flag = SocketWrite(cfd_,
                     metadata->mdata,
                     metadata->total_size * metadata->elsize);
  if (!flag) {
    LAVA_LOG_ERR("TempSkt Send data error\n");
  }
  LAVA_DEBUG(LOG_SKP,
             "Send %ld data\n",
             metadata->total_size * metadata->elsize);
}
void TempSocketSendPort::Join() {
  close(cfd_);
  GetSktManagerSingleton().DeleteSocketFile(addr_path_);
}

TempSocketRecvPort::TempSocketRecvPort(const SocketFile &addr_path) {
  this->name_ = "RecvPort_" + addr_path;
  addr_path_ = addr_path;
  sfd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sfd_ == -1) {
    LAVA_LOG_ERR("Cannot Create Socket Domain File Descripter\n");
  }

  size_t skt_addr_len = sizeof(sa_family_t) + addr_path_.size();
  sockaddr *skt_addr = reinterpret_cast<sockaddr*>(malloc(skt_addr_len));
  skt_addr->sa_family = AF_UNIX;
  memcpy(&skt_addr->sa_data[0], addr_path.c_str(), addr_path_.size());
  // printf("the path: %s, %d\n", &skt_addr->sa_data[0], addr_path_.size());
  if (bind(sfd_, skt_addr, skt_addr_len) == -1) {
    LAVA_LOG_ERR("Cannot bind socket domain\n");
  }
}
void TempSocketRecvPort::Start() {
  if (listen(sfd_, 1) == -1) {
    LAVA_LOG_ERR("Cannot Listen service socket file, %d\n", errno);
  }
}
bool TempSocketRecvPort::Probe() {
  LAVA_LOG_ERR("Not Support TempSocket Port Probe()\n");
  return false;
}
MetaDataPtr TempSocketRecvPort::Recv() {
  bool flag;
  int cfd = accept(sfd_, nullptr, nullptr);
  if (cfd == -1) {
    LAVA_LOG_ERR("Cannot accept the connection\n");
  }
  MetaDataPtr data = std::make_shared<MetaData>();
  flag = SocketRead(cfd, data.get(), sizeof(MetaData));
  if (!flag) {
    LAVA_LOG_ERR("TempSkt Recv data header error\n");
  }
  void *ptr = calloc(data->elsize * data->total_size, 1);
  flag = SocketRead(cfd, ptr, data->elsize * data->total_size);
  if (!flag) {
    LAVA_LOG_ERR("TempSkt Recv data error\n");
  }
  LAVA_DEBUG(LOG_SKP, "Recv %ld data\n", data->elsize * data->total_size);
  data->mdata = ptr;
  close(cfd);
  return data;
}
MetaDataPtr TempSocketRecvPort::Peek() {
  LAVA_LOG_ERR("Not Support TempSocket Port Peek()\n");
  return nullptr;
}
void TempSocketRecvPort::Join() {
  close(sfd_);
  unlink(addr_path_.c_str());
  GetSktManagerSingleton().DeleteSocketFile(addr_path_);
}

}  // namespace message_infrastructure
