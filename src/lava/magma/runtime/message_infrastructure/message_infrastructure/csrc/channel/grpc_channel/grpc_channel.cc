


#include <message_infrastructure/csrc/channel/grpc_channel/grpc_channel.h>
#include <message_infrastructure/csrc/core/utils.h>




namespace message_infrastructure {

GrpcChannel::GrpcChannel(const std::string &src_name,
                           const std::string &dst_name,
                           const size_t &size,
                           const size_t &nbytes) {
  unsigned long data_size = nbytes + sizeof(MetaData);
  
  send_port_ = std::make_shared<GrpcSendPort>(src_name,size, data_size,url);
  recv_port_ = std::make_shared<GrpcRecvPort>(dst_name,size, data_size,url);
}

AbstractSendPortPtr GrpcChannel::GetSendPort() {
  return send_port_;
}

AbstractRecvPortPtr GrpcChannel::GetRecvPort() {
  return recv_port_;
}

std::shared_ptr<GrpcChannel> GetGrpcChannel(const size_t &size,
                              const size_t &nbytes,
                              const std::string &src_name,
                              const std::string &dst_name) {
  return (std::make_shared<GrpcChannel>(src_name,
                                         dst_name,
                                         size,
                                         nbytes));
}
}