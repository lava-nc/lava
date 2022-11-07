cmake_minimum_required(VERSION 3.5)
project(message_passing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O2")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build: Debug, Release" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

option(GRPC_CHANNEL "Use grpc_channel" OFF)
if(GRPC_CHANNEL)
  add_definitions(-DGRPC_CHANNEL)
endif()

option(ENABLE_MM_PAUSE "Use _mm_pause for sleep." OFF)
if(ENABLE_MM_PAUSE)
  add_definitions(-DENABLE_MM_PAUSE)
endif()

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
  "message_infrastructure/csrc/core/channel_factory.cc"
  "message_infrastructure/csrc/core/message_infrastructure_logging.cc"
  "message_infrastructure/csrc/actor/posix_actor.cc"
  "message_infrastructure/csrc/channel/shmem/shm.cc"
  "message_infrastructure/csrc/channel/shmem/shmem_channel.cc"
  "message_infrastructure/csrc/channel/shmem/shmem_port.cc"
  "message_infrastructure/csrc/channel/socket/socket.cc"
  "message_infrastructure/csrc/channel/socket/socket_channel.cc"
  "message_infrastructure/csrc/channel/socket/socket_port.cc")

if(GRPC_CHANNEL)
  set(GRPC_FETCHCONTENT 1)
  set(GRPC_TAG v1.49.1)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/grpc_common.cmake)

  set(grpc_path "${CMAKE_CURRENT_SOURCE_DIR}/message_infrastructure/csrc/channel/grpc/")
  get_filename_component(grpc_proto "${grpc_path}/protos/grpcchannel.proto" ABSOLUTE)
  get_filename_component(grpc_proto_path "${grpc_proto}" PATH)

  set(grpc_proto_srcs "${grpc_path}/grpcchannel.pb.cc")
  set(grpc_proto_hdrs "${grpc_path}/grpcchannel.pb.h")
  set(grpc_srcs "${grpc_path}/grpcchannel.grpc.pb.cc")
  set(grpc_hdrs "${grpc_path}/grpcchannel.grpc.pb.h")
  add_custom_command(
        OUTPUT "${grpc_proto_srcs}" "${grpc_proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${grpc_path}"
             --cpp_out "${grpc_path}"
             -I "${grpc_proto_path}"
             --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
             "${grpc_proto}"
        DEPENDS "${grpc_proto}")

  set(GRPC_CHANNEL_SRCS
    "message_infrastructure/csrc/channel/grpc/grpc.cc"
    "message_infrastructure/csrc/channel/grpc/grpc_port.cc"
    "message_infrastructure/csrc/channel/grpc/grpc_channel.cc"
    ${grpc_proto_srcs}
    ${grpc_srcs})
endif()

option(DDS_CHANNEL "Message library supports DDS Channel" OFF)
option(FASTDDS_ENABLE "enable FastDDS" OFF)
option(CycloneDDS_ENABLE "enable CycloneDDS" OFF)

if(DDS_CHANNEL)
  set(COMMON_DDS_DESTINATION "${CMAKE_BINARY_DIR}/install")
  set(COMMON_DDS_INC "${COMMON_DDS_DESTINATION}/include")
  set(DDS_CHANNEL_SRCS
    "message_infrastructure/csrc/channel/dds/dds.cc"
    "message_infrastructure/csrc/channel/dds/dds_channel.cc")

  if(CycloneDDS_ENABLE)
    set(CycloneDDS_SRC "")
  endif()

  if(FASTDDS_ENABLE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fastdds.cmake)
    add_definitions(-DFASTDDS)
    set(FASTDDS_SRC
      "message_infrastructure/csrc/channel/dds/fast_dds.cc"
      "message_infrastructure/csrc/channel/dds/protos/fast_dds/metadata.cc"
      "message_infrastructure/csrc/channel/dds/protos/fast_dds/metadataPubSubTypes.cc"
      ${DDS_CHANNEL_SRCS})
  endif()

  if(FASTDDS_ENABLE OR CycloneDDS_ENABLE)
    add_definitions(-DDDS_CHANNEL)
  endif()
else()
  set(FASTDDS_ENABLE OFF)
  set(CycloneDDS_ENABLE OFF)
endif()

add_library(message_infrastructure SHARED
  ${MESSAGE_INFRASTRUCTURE_SRCS}
  $<$<BOOL:${GRPC_CHANNEL}>:${GRPC_CHANNEL_SRCS}>
  $<$<BOOL:${FASTDDS_ENABLE}>:${FASTDDS_SRC}>
  $<$<BOOL:${CycloneDDS_ENABLE}>:${CycloneDDS_SRC}>)

set(MSG_LOG_LEVEL err CACHE STRING "Default vaule: err, error log only.")
set(MSG_LOG_FILE_ENABLE 0 CACHE STRING "Default value: 0, print onto console only.")

target_compile_definitions(message_infrastructure PUBLIC
                            $<$<STREQUAL:${MSG_LOG_LEVEL},all>:MSG_LOG_LEVEL_ALL>
                            $<$<STREQUAL:${MSG_LOG_LEVEL},warn>:MSG_LOG_LEVEL_WARN>
                            $<$<STREQUAL:${MSG_LOG_LEVEL},dump>:MSG_LOG_LEVEL_DUMP>
                            $<$<STREQUAL:${MSG_LOG_LEVEL},info>:MSG_LOG_LEVEL_INFO>
                            $<$<STREQUAL:${MSG_LOG_LEVEL},err>:MSG_LOG_LEVEL_ERRO>
                            $<$<STREQUAL:${MSG_LOG_LEVEL},debug>:MSG_LOG_LEVEL_ALL>)

target_compile_definitions(message_infrastructure PUBLIC
                          $<$<EQUAL:${MSG_LOG_FILE_ENABLE},1>:MSG_LOG_FILE_ENABLE>)

target_include_directories(message_infrastructure PUBLIC
                          ${PROJECT_SOURCE_DIR}
                          $<BUILD_INTERFACE:${COMMON_DDS_INC}>)

target_link_libraries(message_infrastructure
                      rt)
if(GRPC_CHANNEL)
  target_link_libraries(message_infrastructure
                        ${_REFLECTION}
                        ${_GRPC_GRPCPP}
                        ${_PROTOBUF_LIBPROTOBUF})
endif()

if(FASTDDS_ENABLE)
  add_dependencies(message_infrastructure foonathan_memory fastcdr fastrtps)
  target_link_libraries(message_infrastructure
    ${COMMON_DDS_DESTINATION}/lib/libfastcdr.so
    ${COMMON_DDS_DESTINATION}/lib/libfastrtps.so)
endif()

if(PY_WRAPPER)
  set(PY_WRAPPER_SRCS
        "message_infrastructure/csrc/message_infrastructure_py_wrapper.cc"
        "message_infrastructure/csrc/channel_proxy.cc"
        "message_infrastructure/csrc/port_proxy.cc")

  pybind11_add_module(MessageInfrastructurePywrapper ${PY_WRAPPER_SRCS})
  target_include_directories(MessageInfrastructurePywrapper PUBLIC
                            ${NUMPY_INCLUDE_DIRS})
  target_link_libraries(MessageInfrastructurePywrapper PRIVATE message_infrastructure)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("debug mode and enable cpp unit test")
  enable_testing()
  add_subdirectory(test)
else()
  message("not debug mode and disable cpp unit test")
endif()