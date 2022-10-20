//#include <numpy/arrayobject.h>
//#include <Python.h>
#include <iostream>
#include <memory>
#include <string>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include "message_infrastructure/csrc/channel/grpc_channel/build/grpcchannel.grpc.pb.h"
#include <message_infrastructure/csrc/channel/grpc_channel/grpc_port.h>
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
:name_(name), size_(size), nbytes_(nbytes), read_index_(0), write_index_(0),done_(false)
{
array_.reserve(size_);
std::cout<<"array_size ===="<<array_.size()<<std::endl;
}

Status GrpcChannelServerImpl::RecvArrayData(ServerContext* context, const GrpcMetaData *request,
						DataReply* reply){
  std::cout<<"recv call 1===="<<std::endl;
	bool ret = false;
  //for(int i =0; i< (nbytes_); i++){
  //      std::cout<<"recv data value = "<<request->value(i)<<std::endl;
  //}
  std::cout<<"recv data value = "<<request->value()<<std::endl;  
  std::cout<<"AviliableCount = "<<AvailableCount()<<std::endl;
	if (AvailableCount() > 0) {
    std::cout<<"if windex="<<write_index_.load()<<std::endl;
    std::cout<<"if rindex="<<read_index_.load()<<std::endl;
		this->Push(request);
		ret=true;
	}
	reply->set_ack(ret);
	return Status::OK;
}

void GrpcChannelServerImpl::Push(const GrpcMetaData *src) {
  auto const curr_write_index = write_index_.load(std::memory_order_relaxed);
  std::cout<<"go push= "<<curr_write_index<<std::endl;
  auto next_write_index = curr_write_index + 1;
  if (next_write_index == size_) {
      next_write_index = 0;
    std::cout<<"go 1 if"<<std::endl;
  }
  if (next_write_index != read_index_.load(std::memory_order_acquire)) {
    std::cout<<"arrar?"<<std::endl;
    array_[curr_write_index] = std::make_shared<GrpcMetaData>(*src);
    std::cout<<"arrar"<<std::endl;
    write_index_.store(next_write_index, std::memory_order_release);
    std::cout<<"stored  write index = "<<write_index_<<std::endl;
  }
  std::cout<<"Push windex = "<<write_index_.load()<<std::endl;
  std::cout<<"Push windex = "<<read_index_.load()<<std::endl;
}

int GrpcChannelServerImpl::AvailableCount() {
	auto const curr_read_index = read_index_.load(std::memory_order_acquire);
	auto const curr_write_index = write_index_.load(std::memory_order_acquire);
	if (curr_read_index == curr_write_index) {
	return size_;
	}
	if (curr_write_index > curr_read_index) {
			return size_ - curr_write_index + curr_read_index - 1;
	}
	return curr_read_index - curr_write_index - 1;
}

bool GrpcChannelServerImpl::Empty() {
	auto const curr_read_index = read_index_.load(std::memory_order_acquire);
	auto const curr_write_index = write_index_.load(std::memory_order_acquire);
	return curr_read_index == curr_write_index;
}

GrpcMetaDataPtr GrpcChannelServerImpl::Pop(bool block) {
	while(block && Empty()) {
    //std::cout<<"Empty() = "<<Empty()<<std::endl;
	  helper::Sleep();
    if(done_)
      return NULL;
	}
	auto const curr_read_index = read_index_.load(std::memory_order_relaxed);
	assert(curr_read_index != write_index_.load(std::memory_order_acquire));
	GrpcMetaDataPtr data_ = array_[curr_read_index];
	auto next_read_index = curr_read_index + 1;
	if(next_read_index == size_) {
			next_read_index = 0;
	}
	read_index_.store(next_read_index, std::memory_order_release);
	return data_;
}


GrpcMetaDataPtr GrpcChannelServerImpl::Front() {
  while(Empty()) {
    helper::Sleep();
    if(done_)
      return NULL;
  }
  auto curr_read_index = read_index_.load(std::memory_order_acquire);
  GrpcMetaDataPtr data_ = array_[curr_read_index];
  return data_;
}

bool GrpcChannelServerImpl::Probe() {
  return !Empty();
}
void GrpcChannelServerImpl::Stop() {
  done_ = true;
}


GrpcRecvPort::GrpcRecvPort(const std::string& name,
                 const size_t &size,
                 const size_t &nbytes, const std::string& url):AbstractRecvPort(name, size, nbytes),done_(false),url_(url){
                    serviceptr = std::make_shared<GrpcChannelServerImpl>(name_, size_, nbytes_);
                 }
void GrpcRecvPort::Startsever(){
  std::cout<<"go?"<<std::endl;
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  builder.AddListeningPort(url_, grpc::InsecureServerCredentials());
  std::cout<<"go?2"<<std::endl;
  builder.RegisterService(serviceptr.get());
  std::cout<<"go3?"<<std::endl;
  server = builder.BuildAndStart();
  std::cout<<"go4?"<<std::endl;
  //std::cout<<"listening to url"<<std::endl;
  server->Wait();

}

void GrpcRecvPort::Start(){
  std::cout<<"recv startthread start"<<std::endl;
  grpcthreadptr = std::make_shared<std::thread>(&message_infrastructure::GrpcRecvPort::Startsever, this);
  std::cout<<"recv startthread end"<<std::endl;
  std::cout<<testtmpsig<<std::endl;
}

MetaDataPtr GrpcRecvPort::Recv(){
  std::cout<<"beging GrpcRcevPort"<<std::endl;
  GrpcMetaDataPtr recvdata = serviceptr->Pop(true);
  std::cout<<"end pop"<<std::endl;
  MetaDataPtr data_ = std::make_shared<MetaData>();
  std::cout<<"111111"<<std::endl;
  GrpcMetaData2MetaData(data_,recvdata);
  std::cout<<"end Recv"<<std::endl;
  return data_;
}

MetaDataPtr GrpcRecvPort::Peek(){
  GrpcMetaDataPtr peekdata = serviceptr->Front();
  MetaDataPtr data_ = std::make_shared<MetaData>();
  GrpcMetaData2MetaData(data_,peekdata);
  return data_;
}

void GrpcRecvPort::Join(){
  if (!done_) {
      done_ = true;
      serviceptr->Stop();
      server->Shutdown();
      grpcthreadptr->join();
  }
}

bool GrpcRecvPort::Probe(){
  serviceptr->Probe();
}
void GrpcRecvPort::GrpcMetaData2MetaData(MetaDataPtr metadata, GrpcMetaDataPtr grpcdata){
  std::cout<<"begin GrpcMetaData2MetaData"<<std::endl;
  metadata->nd = grpcdata->nd();
  metadata->type = grpcdata->type();
  metadata->elsize = grpcdata->elsize();
  metadata->total_size = grpcdata->total_size();
  std::cout<<"2346543154"<<std::endl;
  void* data =malloc(nbytes_);
  
  for(int i=0; i<MAX_ARRAY_DIMS;i++){
          metadata->dims[i] = grpcdata->dims(i);
          metadata->strides[i] = grpcdata->strides(i);
  }
  std::cout<<"2======"<<std::endl;
  std::cout<<"G2M"<<grpcdata->value()<<std::endl;
  std::memcpy(data, grpcdata->value().c_str(), nbytes_);
  metadata->mdata = data;
  //for(int i =0; i< (nbytes_); i++){
  //        *(data+i)=grpcdata->value(i);
  //}
  std::cout<<"end GrpcMetaData2MetaData"<<std::endl;
}



GrpcSendPort::GrpcSendPort(const std::string &name,
                const size_t &size,
                const size_t &nbytes, const std::string& url)
  :AbstractSendPort(name, size, nbytes),done_(false),url_(url)
{}

void GrpcSendPort::Start(){
    channel = grpc::CreateChannel(url_,grpc::InsecureChannelCredentials());
    stub_ = GrpcChannelServer::NewStub(channel);
}

GrpcMetaData GrpcSendPort::MetaData2GrpcMetaData(MetaDataPtr metadata){
  GrpcMetaData grpcdata;
  printf("send metadata_nd=%d\n",metadata->nd);
  grpcdata.set_nd(metadata->nd);
  printf("send grpcmetadata_nd=%d\n",grpcdata.nd());
  grpcdata.set_type(metadata->type);
  printf("send grpcmetadata_type=%d\n",grpcdata.type());
  grpcdata.set_elsize(metadata->elsize);
  printf("send grpcmetadata_elsize=%d\n",grpcdata.elsize());
  grpcdata.set_total_size(metadata->total_size);
  printf("send metadata_total_size=%d\n",grpcdata.total_size());
  

  char* data = (char*)metadata->mdata;
  //std::cout<<"send metadata = "<<std::endl;
  //std::cout<<data<<std::endl;
  for(int i=0; i<MAX_ARRAY_DIMS;i++){
          grpcdata.add_dims(metadata->dims[i]);
          grpcdata.add_strides(metadata->strides[i]);
  }
  grpcdata.set_value(data);
  //std::cout<<"send grpcdata = "<<std::endl;
  //std::cout<<grpcdata.value()<<std::endl;
  //for(int i =0; i< (nbytes_); i++){
  //        std::cout<<"send metadata value = "<<*(data+i)<<std::endl;
  //        
  //        std::cout<<"send grpcmetadata value = "<<grpcdata.value(i)<<std::endl;
  //}
  return grpcdata;
}

void GrpcSendPort::Send(MetaDataPtr metadata){
  GrpcMetaData request = MetaData2GrpcMetaData(metadata);
  DataReply reply;
  ClientContext context;
  Status status = stub_->RecvArrayData(&context, request, &reply);
  printf("=======\n");
  printf("stub send");

  if (status.ok()&&reply.ack()) {
      std::cout<<"Successed,Recv ack is "<<reply.ack()<<std::endl;
  } else {
      std::cout<<"Recv ack is "<<reply.ack()<<"ERROR! Send fail!"<<std::endl;
  }
}
bool GrpcSendPort::Probe() {
  return false;
}
void GrpcSendPort::Join() {
  done_ = true;
}

}