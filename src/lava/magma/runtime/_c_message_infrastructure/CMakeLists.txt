cmake_minimum_required(VERSION 3.5)
project(message_passing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O2")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build: Debug, Release" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

option(GRPC_CHANNEL "Use grpc_channel" OFF)
option(DDS_CHANNEL "Message library supports DDS Channel" OFF)
option(FASTDDS_ENABLE "enable FastDDS" OFF)
option(CycloneDDS_ENABLE "enable CycloneDDS" OFF)

if(GRPC_CHANNEL AND CycloneDDS_ENABLE)
  message(FATAL_ERROR "Cannot enable GRPC and CycloneDDS together")
endif()

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
  "csrc/core/abstract_actor.cc"
  "csrc/core/abstract_port.cc"
  "csrc/core/multiprocessing.cc"
  "csrc/core/abstract_port_implementation.cc"
  "csrc/core/ports.cc"
  "csrc/core/channel_factory.cc"
  "csrc/core/message_infrastructure_logging.cc"
  "csrc/actor/posix_actor.cc"
  "csrc/channel/shmem/shm.cc"
  "csrc/channel/shmem/shmem_channel.cc"
  "csrc/channel/shmem/shmem_port.cc"
  "csrc/channel/socket/socket.cc"
  "csrc/channel/socket/socket_channel.cc"
  "csrc/channel/socket/socket_port.cc")

if(GRPC_CHANNEL)
  set(GRPC_FETCHCONTENT 1)
  set(GRPC_TAG v1.49.1)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/grpc_common.cmake)

  set(grpc_path "${CMAKE_CURRENT_SOURCE_DIR}/csrc/channel/grpc/")
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
    "csrc/channel/grpc/grpc.cc"
    "csrc/channel/grpc/grpc_port.cc"
    "csrc/channel/grpc/grpc_channel.cc"
    ${grpc_proto_srcs}
    ${grpc_srcs})
endif()


if(DDS_CHANNEL)
  set(COMMON_DDS_DESTINATION "${CMAKE_BINARY_DIR}/install")
  set(COMMON_DDS_INC "${COMMON_DDS_DESTINATION}/include")
  set(DDS_CHANNEL_SRCS
    "csrc/channel/dds/dds.cc"
    "csrc/channel/dds/dds_channel.cc")

  if(FASTDDS_ENABLE AND CycloneDDS_ENABLE)
	  message(FATAL_ERROR "Cannot enable Backend FASTDDS and CycloneDDS together")
  endif()

  if(CycloneDDS_ENABLE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cyclonedds.cmake)
    set(CYCLONE_DDS_INC "${COMMON_DDS_DESTINATION}/include/ddscxx")
    add_custom_target(CycloneDDS_metdata ALL
                      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/channel/dds/protos/cyclone_dds/DDSMetaData.cpp")
    add_dependencies(CycloneDDS_metdata cyclonedds-cxx)
    add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/csrc/channel/dds/protos/cyclone_dds/DDSMetaData.cpp"
                       COMMAND "${COMMON_DDS_DESTINATION}/bin/idlc" -l cxx ${CMAKE_CURRENT_SOURCE_DIR}/csrc/channel/dds/protos/DDSMetaData.idl
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/csrc/channel/dds/protos/cyclone_dds)

    add_definitions(-DCycloneDDS_ENABLE)
    set(CycloneDDS_SRC
      ${DDS_CHANNEL_SRCS}
      "csrc/channel/dds/cyclone_dds.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/channel/dds/protos/cyclone_dds/DDSMetaData.cpp")
  endif()

  if(FASTDDS_ENABLE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fastdds.cmake)
    add_definitions(-DFASTDDS_ENABLE)
    set(FASTDDS_SRC
      ${DDS_CHANNEL_SRCS}
      "csrc/channel/dds/fast_dds.cc"
      "csrc/channel/dds/protos/fast_dds/DDSMetaData.cc"
      "csrc/channel/dds/protos/fast_dds/DDSMetaDataPubSubTypes.cc")
  endif()

  if(FASTDDS_ENABLE OR CycloneDDS_ENABLE)
    add_definitions(-DDDS_CHANNEL)
  else()
    message(FATAL_ERROR "Please enable DDS backend, (FASTDDS_ENABLE or CycloneDDS_ENABLE)")
  endif()
else()
  set(FASTDDS_ENABLE OFF)
  set(CycloneDDS_ENABLE OFF)
endif()

add_library(message_infrastructure SHARED
  ${MESSAGE_INFRASTRUCTURE_SRCS}
  $<$<BOOL:${CycloneDDS_ENABLE}>:${CycloneDDS_SRC}>
  $<$<BOOL:${FASTDDS_ENABLE}>:${FASTDDS_SRC}>
  $<$<BOOL:${GRPC_CHANNEL}>:${GRPC_CHANNEL_SRCS}>)

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
                          ${PROJECT_SOURCE_DIR}/csrc
                          $<BUILD_INTERFACE:${COMMON_DDS_INC}>
                          $<BUILD_INTERFACE:${CYCLONE_DDS_INC}>)

target_link_libraries(message_infrastructure
                      rt)
if(GRPC_CHANNEL)
  target_link_libraries(message_infrastructure
                        ${_REFLECTION}
                        ${_GRPC_GRPCPP}
                        ${_PROTOBUF_LIBPROTOBUF})
endif()

if(CycloneDDS_ENABLE)
  add_dependencies(message_infrastructure cyclonedds-cxx CycloneDDS_metdata)
  target_link_libraries(message_infrastructure
    ${COMMON_DDS_DESTINATION}/lib/libddsc.so
    ${COMMON_DDS_DESTINATION}/lib/libddscxx.so)
endif()

if(FASTDDS_ENABLE)
  add_dependencies(message_infrastructure foonathan_memory fastcdr fastrtps)
  target_link_libraries(message_infrastructure
    ${COMMON_DDS_DESTINATION}/lib/libfastcdr.so
    ${COMMON_DDS_DESTINATION}/lib/libfastrtps.so)
endif()

if(PY_WRAPPER)
  set(PY_WRAPPER_SRCS
        "csrc/message_infrastructure_py_wrapper.cc"
        "csrc/channel_proxy.cc"
        "csrc/port_proxy.cc")

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
