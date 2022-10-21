cmake_minimum_required(VERSION 3.5.1)

project(GrpcChannel)

include(protos/common.cmake)

# Proto file
get_filename_component(grpc_proto "protos/grpcchannel.proto" ABSOLUTE)
get_filename_component(grpc_proto_path "${grpc_proto}" PATH)

# Generated sources
set(grpc_proto_srcs "grpcchannel.pb.cc")
set(grpc_proto_hdrs "grpcchannel.pb.h")
set(grpc_srcs "grpcchannel.grpc.pb.cc")
set(grpc_hdrs "grpcchannel.grpc.pb.h")
add_custom_command(
      OUTPUT "${grpc_proto_srcs}" "${grpc_proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "./"
        --cpp_out "./"
        -I "${grpc_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${grpc_proto}"
      DEPENDS "${grpc_proto}")

# Include generated *.pb.h files
include_directories("./")

add_library(grpc_proto
  ${grpc_srcs}
  ${grpc_hdrs}
  ${grpc_proto_srcs}
  ${grpc_proto_hdrs})
target_link_libraries(grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})