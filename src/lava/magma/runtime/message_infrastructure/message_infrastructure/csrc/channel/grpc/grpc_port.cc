// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <message_infrastructure/csrc/channel/grpc/grpc_port.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <message_infrastructure/csrc/channel/grpc/build/grpcchannel.grpc.pb.h>

#include <memory>
#include <iostream>
#include <string>

namespace message_infrastructure {

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpcchannel::DataReply;
using grpcchannel::GrpcChannelServer;
using grpcchannel::GrpcMetaData;
using GrpcMetaDataPtr = std::shared_ptr<GrpcMetaData>;

GrpcChannelServerImpl::GrpcChannelServerImpl(const std::string& name,
                                             const size_t &size)
                                             :name_(name),
                                             size_(size),
                                             done_(false) {
  recv_queue_ = std::make_shared<RecvQueue<GrpcMetaDataPtr>>(name_, size_);
}
Status GrpcChannelServerImpl::RecvArrayData(ServerContext* context,
                                            const GrpcMetaData *request,
                                            DataReply* reply) {
  bool rep = true;
  while (recv_queue_->AvailableCount() <=0) {
    helper::Sleep();
    if (done_) {
      rep = false;
      return Status::OK;
    }
  }
  recv_queue_->Push(std::make_shared<GrpcMetaData>(*request));
  reply->set_ack(rep);
  return Status::OK;
}
GrpcMetaDataPtr GrpcChannelServerImpl::Pop(bool block) {
  return recv_queue_->Pop(block);
}
GrpcMetaDataPtr GrpcChannelServerImpl::Front() {
  return recv_queue_->Front();
}
bool GrpcChannelServerImpl::Probe() {
  return recv_queue_->Probe();
}
void GrpcChannelServerImpl::Stop() {
  done_ = true;
  recv_queue_->Stop();
}

GrpcRecvPort::GrpcRecvPort(const std::string& name,
                           const size_t &size,
                           const std::string& url)
                           :name_(name),
                           size_(size),
                           done_(false),
                           url_(url) {
  serviceptr = std::make_shared<GrpcChannelServerImpl>(name_, size_);
}
void GrpcRecvPort::Start() {
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  builder_.AddListeningPort(url_, grpc::InsecureServerCredentials());
  builder_.RegisterService(serviceptr.get());
  server_ = builder_.BuildAndStart();
}
MetaDataPtr GrpcRecvPort::Recv() {
  GrpcMetaDataPtr recvdata = serviceptr->Pop(true);
  MetaDataPtr data_ = std::make_shared<MetaData>();
  GrpcMetaData2MetaData(data_, recvdata);
  return data_;
}
MetaDataPtr GrpcRecvPort::Peek() {
  GrpcMetaDataPtr peekdata = serviceptr->Front();
  MetaDataPtr data_ = std::make_shared<MetaData>();
  GrpcMetaData2MetaData(data_, peekdata);
  return data_;
}
void GrpcRecvPort::Join() {
  if (!done_) {
    done_ = true;
    serviceptr->Stop();
    server_->Shutdown();
  }
}
bool GrpcRecvPort::Probe() {
  return serviceptr->Probe();
}
void GrpcRecvPort::GrpcMetaData2MetaData(MetaDataPtr metadata,
                                         GrpcMetaDataPtr grpcdata) {
  metadata->nd = grpcdata->nd();
  metadata->type = grpcdata->type();
  metadata->elsize = grpcdata->elsize();
  metadata->total_size = grpcdata->total_size();
  void* data = malloc(metadata->elsize*metadata->total_size);
  for (int i = 0; i < MAX_ARRAY_DIMS; i++) {
    metadata->dims[i] = grpcdata->dims(i);
    metadata->strides[i] = grpcdata->strides(i);
  }
  std::memcpy(data,
              grpcdata->value().c_str(),
              metadata->elsize*metadata->total_size);
  metadata->mdata = data;
}


void GrpcSendPort::Start() {
  channel_ = grpc::CreateChannel(url_, grpc::InsecureChannelCredentials());
  stub_ = GrpcChannelServer::NewStub(channel_);
}
GrpcMetaData GrpcSendPort::MetaData2GrpcMetaData(MetaDataPtr metadata) {
  GrpcMetaData grpcdata;
  grpcdata.set_nd(metadata->nd);
  grpcdata.set_type(metadata->type);
  grpcdata.set_elsize(metadata->elsize);
  grpcdata.set_total_size(metadata->total_size);
  char* data = reinterpret_cast<char*>(metadata->mdata);
  for (int i = 0; i < MAX_ARRAY_DIMS; i++) {
    grpcdata.add_dims(metadata->dims[i]);
    grpcdata.add_strides(metadata->strides[i]);
  }
  grpcdata.set_value(data, metadata->elsize*metadata->total_size);
  return grpcdata;
}
void GrpcSendPort::Send(MetaDataPtr metadata) {
  GrpcMetaData request = MetaData2GrpcMetaData(metadata);
  DataReply reply;
  ClientContext context;
  Status status = stub_->RecvArrayData(&context, request, &reply);
  if (!reply.ack()) {
    LAVA_LOG_ERR("Send fail!");
  }
}
bool GrpcSendPort::Probe() {
  return false;
}
void GrpcSendPort::Join() {
  done_ = true;
}

}  // namespace message_infrastructure
