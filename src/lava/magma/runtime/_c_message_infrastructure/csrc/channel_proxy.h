// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_PROXY_H_
#define CHANNEL_PROXY_H_

#include <core/abstract_channel.h>
#include <core/utils.h>
#include <port_proxy.h>
#if defined(DDS_CHANNEL)
#include <channel/dds/dds.h>
#endif

#include <string>
#include <vector>

namespace message_infrastructure {

class ChannelProxy {
 public:
  ChannelProxy(const ChannelType &channel_type,
               const size_t &size,
               const size_t &nbytes,
               const std::string &src_name,
               const std::string &dst_name,
               py::tuple shape = py::make_tuple(),
               py::object type = py::none());
  SendPortProxyPtr GetSendPort();
  RecvPortProxyPtr GetRecvPort();
 private:
  AbstractChannelPtr channel_ = nullptr;
  SendPortProxyPtr send_port_ = nullptr;
  RecvPortProxyPtr recv_port_ = nullptr;
};

class TempChannelProxy {
 public:
  TempChannelProxy();
  explicit TempChannelProxy(const std::string &addr_path);
  SendPortProxyPtr GetSendPort();
  RecvPortProxyPtr GetRecvPort();
  std::string GetAddrPath();
 private:
  AbstractChannelPtr channel_ = nullptr;
};

#if defined(GRPC_CHANNEL)
class GetRPCChannelProxy {
 public:
  GetRPCChannelProxy(const std::string &url,
                     const int &port,
                     const std::string &src_name,
                     const std::string &dst_name,
                     const size_t &size);
  GetRPCChannelProxy(const std::string &src_name,
                     const std::string &dst_name,
                     const size_t &size);
  SendPortProxyPtr GetSendPort();
  RecvPortProxyPtr GetRecvPort();
 private:
  ChannelType channel_type = ChannelType::RPCCHANNEL;
  AbstractChannelPtr channel_ = nullptr;
  SendPortProxyPtr send_port_ = nullptr;
  RecvPortProxyPtr recv_port_ = nullptr;
};
#endif

#if defined(DDS_CHANNEL)
class GetDDSChannelProxy {
 public:
  GetDDSChannelProxy(const std::string &src_name,
                     const std::string &dst_name,
                     const size_t &size,
                     const size_t &nbytes,
                     const DDSTransportType &dds_transfer_type,
                     const DDSBackendType &dds_backend);
  SendPortProxyPtr GetSendPort();
  RecvPortProxyPtr GetRecvPort();
 private:
  AbstractChannelPtr channel_ = nullptr;
  SendPortProxyPtr send_port_ = nullptr;
  RecvPortProxyPtr recv_port_ = nullptr;
};
#endif

}  // namespace message_infrastructure

#endif  // CHANNEL_PROXY_H_
