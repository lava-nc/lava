#include <message_infrastructure/csrc/channel/grpc_channel/grpc.h>

namespace message_infrastructure {

GrpcManager::~GrpcManager() {
  port_num = 0;
	url_num = 0;
  urls_.clear();
}

GrpcManager GrpcManager::grpcm_;

GrpcManager& GetGrpcManager(){
  GrpcManager &grpcm = GrpcManager::grpcm_;
  return grpcm;
}
}  // namespace message_infrastructure
