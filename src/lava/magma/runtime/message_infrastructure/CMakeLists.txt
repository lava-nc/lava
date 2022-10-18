cmake_minimum_required(VERSION 3.5.1)
project(message_passing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("message_infrastructure/csrc/channel/grpc_channel")
# # Proto file
# get_filename_component(hw_proto "message_infrastructure/csrc/channel/grpc_channel/protos/grpcchannel.proto" ABSOLUTE)
# get_filename_component(hw_proto_path "${hw_proto}" PATH)

# # Generated sources
# set(grpc_root "message_infrastructure/csrc/channel/grpc_channel")
# set(hw_proto_srcs "${grpc_root}/grpcchannel.pb.cc")
# set(hw_proto_hdrs "${grpc_root}/grpcchannel.pb.h")
# set(hw_grpc_srcs "${grpc_root}/grpcchannel.grpc.pb.cc")
# set(hw_grpc_hdrs "${grpc_root}/grpcchannel.grpc.pb.h")

# # Include generated *.pb.h files
# include_directories("${grpc_root}")

# # hw_grpc_proto
# add_library(hw_grpc_proto
#   ${hw_grpc_srcs}
#   ${hw_grpc_hdrs}
#   ${hw_proto_srcs}
#   ${hw_proto_hdrs})
# target_link_libraries(hw_grpc_proto
#   ${_REFLECTION}
#   ${_GRPC_GRPCPP}
#   ${_PROTOBUF_LIBPROTOBUF})

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
                      hw_grpc_proto
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
    target_link_libraries(MessageInfrastructurePywrapper PRIVATE message_infrastructure)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("debug mode and enable cpp unit test")
    enable_testing()
    add_subdirectory(test)
else()
    message("not debug mode and disable cpp unit test")
endif()
