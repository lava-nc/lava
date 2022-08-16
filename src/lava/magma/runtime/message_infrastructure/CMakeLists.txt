cmake_minimum_required(VERSION 3.5)
project(message_passing)

set (CMAKE_CXX_STANDARD 17)
include_directories(message_infrastructure/csrc)

find_package(pybind11 REQUIRED)

file (GLOB MULTI_PROC_SRCS "message_infrastructure/csrc/multiprocessing.cc"
                           "message_infrastructure/csrc/shmem_port.cc")

file (GLOB PY_WRAPPER "message_infrastructure/csrc/message_infrastructure_py_wrapper.cc")

add_library(multi_processing SHARED ${MULTI_PROC_SRCS})
target_include_directories(multi_processing PUBLIC 
                           ${pybind11_INCLUDE_DIRS}
                           ${PROJECT_SOURCE_DIR}/message_infrastructure/csrc
                           )

pybind11_add_module(MessageInfrastructurePywrapper ${PY_WRAPPER})
target_link_libraries(MessageInfrastructurePywrapper PRIVATE multi_processing)
