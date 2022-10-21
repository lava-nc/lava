# Copyright 2018 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cmake build file for C++ helloworld example.
# Assumes protobuf and gRPC have been installed using cmake.
# See cmake_externalproject/CMakeLists.txt for all-in-one cmake build
# that automatically builds all the dependencies before building helloworld.

cmake_minimum_required(VERSION 3.5.1)

project(GrpcChannel)

include(cmake/common.cmake)

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

# hw_grpc_proto
add_library(grpc_proto
  ${grpc_srcs}
  ${grpc_hdrs}
  ${grpc_proto_srcs}
  ${grpc_proto_hdrs})
target_link_libraries(grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})