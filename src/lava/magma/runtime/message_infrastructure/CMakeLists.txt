cmake_minimum_required(VERSION 2.8.3)
project(message_passing)

set (CMAKE_CXX_STANDARD 17)
include_directories(message_infrastructure/csrc)

set(MULTI_PROC_SRCS 
    "message_infrastructure/csrc/multiprocessing.cc"
    "message_infrastructure/csrc/posix_actor.cc"
)

set(ABSTRACT_PORT_SRCS
    "message_infrastructure/csrc/abstract_port_implementation.cc"
)

set(PORT_SRCS
    "message_infrastructure/csrc/ports.cc"
)

set(TRANSFORMER_SRCS
    "message_infrastructure/csrc/transformer.cc"
)

file (GLOB PY_WRAPPER "message_infrastructure/csrc/message_infrastructure_py_wrapper.cc")

find_package(pybind11 CONFIG)

add_library(multi_processing SHARED ${MULTI_PROC_SRCS})

pybind11_add_module(MessageInfrastructurePywrapper ${PY_WRAPPER})
target_link_libraries(MessageInfrastructurePywrapper PRIVATE multi_processing)
