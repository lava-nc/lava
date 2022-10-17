cmake_minimum_required(VERSION 3.5.1)
project(message_passing)

include(message_infrastructure/csrc/channel/grpc_channel/cmake/common.cmake)

get_filename_component(grpc_channel_proto "./message_infrastructure/csrc/channel/grpc_channel/protos/grpcchannel.proto" ABSOLUTE)
get_filename_component(grpc_channel_proto_path "${grpc_channel_proto}" PATH)

set(grpc_channel_proto_srcs "message_infrastructure/csrc/channel/grpc_channel/grpcchannel.pb.cc")
set(grpc_channel_proto_hdrs "message_infrastructure/csrc/channel/grpc_channel/grpcchannel.pb.h")
set(grpc_channel_grpc_srcs "message_infrastructure/csrc/channel/grpc_channel/grpcchannel.grpc.pb.cc")
set(grpc_channel_grpc_hdrs "message_infrastructure/csrc/channel/grpc_channel/grpcchannel.grpc.pb.h")

add_custom_command(
      OUTPUT "${grpc_channel_proto_srcs}" "${grpc_channel_proto_hdrs}" "${grpc_channel_grpc_srcs}" "${grpc_channel_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "message_infrastructure/csrc/channel/grpc_channel"
        --cpp_out "message_infrastructure/csrc/channel/grpc_channel"
        -I "${grpc_channel_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${grpc_channel_proto}"
      DEPENDS "${grpc_channel_proto}")

include_directories("message_infrastructure/csrc/channel/grpc_channel")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PY_WRAPPER "Use pybind11 to wrapper the message infrastructure lib" ON)
if(PY_WRAPPER)
    find_package(pybind11 REQUIRED)
endif()

set(MESSAGE_INFRASTRUCTURE_SRCS
    "message_infrastructure/csrc/core/abstract_actor.cc"
    "message_infrastructure/csrc/core/abstract_port.cc"
    "message_infrastructure/csrc/core/multiprocessing.cc"
    "message_infrastructure/csrc/core/abstract_port_implementation.cc"
    "message_infrastructure/csrc/core/ports.cc"
    "message_infrastructure/csrc/actor/posix_actor.cc"
    "message_infrastructure/csrc/channel/shmem/shm.cc"
    "message_infrastructure/csrc/channel/shmem/shmem_channel.cc"
    "message_infrastructure/csrc/channel/shmem/shmem_port.cc"
    "message_infrastructure/csrc/channel/grpc_channel/grpc_channel.cc"
    "message_infrastructure/csrc/channel/grpc_channel/grpcchannel.pb.cc"
    "message_infrastructure/csrc/channel/grpc_channel/grpcchannel.grpc.pb.cc"
    "message_infrastructure/csrc/channel/socket/socket.cc"
    "message_infrastructure/csrc/channel/socket/socket_channel.cc"
    "message_infrastructure/csrc/channel/socket/socket_port.cc"

    )

add_library(message_infrastructure SHARED ${MESSAGE_INFRASTRUCTURE_SRCS})

target_include_directories(message_infrastructure PUBLIC
                           ${PROJECT_SOURCE_DIR}
                           )

target_link_libraries(message_infrastructure
                      rt
                      )

if(PY_WRAPPER)
    set(PY_WRAPPER_SRCS
        "message_infrastructure/csrc/message_infrastructure_py_wrapper.cc"
        "message_infrastructure/csrc/channel_proxy.cc"
        "message_infrastructure/csrc/port_proxy.cc"
        )

    pybind11_add_module(MessageInfrastructurePywrapper ${PY_WRAPPER_SRCS})
    target_include_directories(MessageInfrastructurePywrapper PUBLIC
                               ${NUMPY_INCLUDE_DIRS}
                               )
    target_link_libraries(MessageInfrastructurePywrapper PRIVATE message_infrastructure 
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("debug mode and enable cpp unit test")
    enable_testing()
    add_subdirectory(test)
else()
    message("not debug mode and disable cpp unit test")
endif()
