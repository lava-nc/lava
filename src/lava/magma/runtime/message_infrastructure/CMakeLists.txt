cmake_minimum_required(VERSION 3.5)
project(message_passing)

set (CMAKE_CXX_STANDARD 17)

option(USE_PYBIND11 "Use pybind11 to wrapper the message infrastructure lib" ON)
if(USE_PYBIND11)
    find_package(pybind11 REQUIRED)
endif()

set (MESSAGE_INFRASTRUCTURE_SRCS
     "message_infrastructure/csrc/multiprocessing.cc"
     "message_infrastructure/csrc/posix_actor.cc"
     "message_infrastructure/csrc/shmem_port.cc"
     "message_infrastructure/csrc/abstract_port_implementation.cc"
     "message_infrastructure/csrc/ports.cc"
     "message_infrastructure/csrc/transformer.cc"
     )

file (GLOB PY_WRAPPER "message_infrastructure/csrc/message_infrastructure_py_wrapper.cc")

add_library(message_infrastructure SHARED ${MESSAGE_INFRASTRUCTURE_SRCS})

if(USE_PYBIND11)
    target_include_directories(message_infrastructure PUBLIC 
                               ${pybind11_INCLUDE_DIRS}
                               ${PROJECT_SOURCE_DIR}/message_infrastructure/csrc
                               )

    pybind11_add_module(MessageInfrastructurePywrapper ${PY_WRAPPER})
    target_link_libraries(MessageInfrastructurePywrapper PRIVATE message_infrastructure)
else()
    target_include_directories(message_infrastructure PUBLIC 
                               ${PROJECT_SOURCE_DIR}/message_infrastructure/csrc
                               )
endif()
