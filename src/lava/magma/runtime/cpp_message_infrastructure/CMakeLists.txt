cmake_minimum_required(VERSION 2.8.3)
project(message_passing)

set (CMAKE_CXX_STANDARD 17)
include_directories(include)

file (GLOB MULTI_PROC_SRCS "src/multiprocessing.cc")

file (GLOB PY_WRAPPER "src/py_wrapper.cc")

add_subdirectory(pybind11)

add_library(multi_processing SHARED ${MULTI_PROC_SRCS})

pybind11_add_module(MessageInfrastructurePywrapper ${PY_WRAPPER})
target_link_libraries(MessageInfrastructurePywrapper PRIVATE multi_processing)
