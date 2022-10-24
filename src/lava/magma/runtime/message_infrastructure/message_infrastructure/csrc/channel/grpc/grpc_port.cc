// Copyright (C) 2021 Intel Corporation
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
                                             const size_t &size,
                                             const size_t &nbytes)
                        :name_(name), size_(size),
                        nbytes_(nbytes), done_(false) {
  recvqueue = std::make_shared<RecvQueue<GrpcMetaDataPtr>>(name_,
                                                           size_,
                                                           nbytes_);
}
Status GrpcChannelServerImpl::RecvArrayData(ServerContext* context,
                                            const GrpcMetaData *request,
                                            DataReply* reply) {
  bool rep = true;
  while (recvqueue->AvailableCount() <=0) {
    helper::Sleep();
    if (done_) {
      rep = false;
      return Status::OK;
    }
  }
  recvqueue->Push(std::make_shared<GrpcMetaData>(*request));
  reply->set_ack(rep);
  return Status::OK;
}
GrpcMetaDataPtr GrpcChannelServerImpl::Pop(bool block) {
  return recvqueue->Pop(block);
}
GrpcMetaDataPtr GrpcChannelServerImpl::Front() {
  return recvqueue->Front();
}
bool GrpcChannelServerImpl::Probe() {
  return recvqueue->Probe();
}
void GrpcChannelServerImpl::Stop() {
  done_ = true;
  recvqueue->Stop();
}


GrpcRecvPort::GrpcRecvPort(const std::string& name,
                 const size_t &size,
                 const size_t &nbytes, const std::string& url)
                 :AbstractRecvPort(name, size, nbytes),
                 done_(false),
                 url_(url) {
                  serviceptr = std::make_shared<GrpcChannelServerImpl>(name_,
                                                                       size_,
                                                                      nbytes_);
                 }
GrpcRecvPort::~GrpcRecvPort() {
  // have bug ready to be fix
}
void GrpcRecvPort::Start() {
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  builder.AddListeningPort(url_, grpc::InsecureServerCredentials());
  builder.RegisterService(serviceptr.get());
  server_ = builder.BuildAndStart();
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
  void* data = malloc(nbytes_);
  for (int i = 0; i < MAX_ARRAY_DIMS; i++) {
    metadata->dims[i] = grpcdata->dims(i);
    metadata->strides[i] = grpcdata->strides(i);
  }
  std::memcpy(data, grpcdata->value().c_str(), nbytes_);
  metadata->mdata = data;
}

GrpcSendPort::GrpcSendPort(const std::string &name,
                           const size_t &size,
                           const size_t &nbytes, const std::string& url)
  :AbstractSendPort(name, size, nbytes), done_(false), url_(url) {}
void GrpcSendPort::Start() {
  channel = grpc::CreateChannel(url_, grpc::InsecureChannelCredentials());
  stub_ = GrpcChannelServer::NewStub(channel);
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
  grpcdata.set_value(data, nbytes_);
  return grpcdata;
}
void GrpcSendPort::Send(MetaDataPtr metadata) {
  GrpcMetaData request = MetaData2GrpcMetaData(metadata);
  DataReply reply;
  ClientContext context;
  Status status = stub_->RecvArrayData(&context, request, &reply);
}
bool GrpcSendPort::Probe() {
  return false;
}
void GrpcSendPort::Join() {
  done_ = true;
}

}  // namespace message_infrastructure
