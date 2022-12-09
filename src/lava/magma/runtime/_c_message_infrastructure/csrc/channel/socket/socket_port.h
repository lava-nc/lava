// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_SOCKET_SOCKET_PORT_H_
#define CHANNEL_SOCKET_SOCKET_PORT_H_

#include <channel/socket/socket.h>
#include <core/utils.h>
#include <core/abstract_port.h>

#include <string>
#include <vector>
#include <memory>

namespace message_infrastructure {

class SocketSendPort final : public AbstractSendPort {
 public:
  SocketSendPort() = delete;
  SocketSendPort(const std::string &name,
                 const SocketPair &socket,
                 const size_t &nbytes) :
                 name_(name), nbytes_(nbytes), socket_(socket) {}
  ~SocketSendPort() override {}
  void Start();
  void Send(DataPtr metadata);
  void Join();
  bool Probe();

 private:
  std::string name_;
  size_t nbytes_;
  SocketPair socket_;
};

// Users should be allowed to copy port objects.
// Use std::shared_ptr.
using SocketSendPortPtr = std::shared_ptr<SocketSendPort>;

class SocketRecvPort final : public AbstractRecvPort {
 public:
  SocketRecvPort() = delete;
  SocketRecvPort(const std::string &name,
                 const SocketPair &socket,
                 const size_t &nbytes) :
                 name_(name), nbytes_(nbytes), socket_(socket) {}
  ~SocketRecvPort() override {}
  void Start();
  bool Probe();
  MetaDataPtr Recv();
  void Join();
  MetaDataPtr Peek();

 private:
  std::string name_;
  size_t nbytes_;
  SocketPair socket_;
};

// Users should be allowed to copy port objects.
// Use std::shared_ptr.
using SocketRecvPortPtr = std::shared_ptr<SocketRecvPort>;

class TempSocketSendPort final : public AbstractSendPort {
 public:
  TempSocketSendPort() = delete;
  TempSocketSendPort(const SocketFile &addr_path);
  ~TempSocketSendPort() override {};
  void Start();
  void Send(DataPtr metadata);
  void Join();
  bool Probe();
 private:
  int cfd_;
  SocketFile addr_path_;
};
using TempSocketSendPortPtr = std::shared_ptr<TempSocketSendPort>;

class TempSocketRecvPort final : public AbstractRecvPort {
 public:
  TempSocketRecvPort() = delete;
  TempSocketRecvPort(const SocketFile &addr_path);
  ~TempSocketRecvPort() override {}
  void Start();
  bool Probe();
  void Join();
  MetaDataPtr Recv();
  MetaDataPtr Peek();
 private:
  int sfd_;
  SocketFile addr_path_;
};
using TempSocketRecvPortPtr = std::shared_ptr<TempSocketRecvPort>;

}  // namespace message_infrastructure

#endif  // CHANNEL_SOCKET_SOCKET_PORT_H_
