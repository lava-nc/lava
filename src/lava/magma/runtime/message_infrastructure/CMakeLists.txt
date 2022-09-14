cmake_minimum_required(VERSION 3.5)
project(message_passing)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lrt")

option(PY_WRAPPER "Use pybind11 to wrapper the message infrastructure lib" ON)
if(PY_WRAPPER)
    find_package(pybind11 REQUIRED)
endif()

set (MESSAGE_INFRASTRUCTURE_SRCS
     "message_infrastructure/csrc/abstract_actor.cc"
     "message_infrastructure/csrc/abstract_port.cc"
     "message_infrastructure/csrc/multiprocessing.cc"
     "message_infrastructure/csrc/posix_actor.cc"
     "message_infrastructure/csrc/shm.cc"
     "message_infrastructure/csrc/shmem_channel.cc"
     "message_infrastructure/csrc/shmem_port.cc"
     )

add_library(message_infrastructure SHARED ${MESSAGE_INFRASTRUCTURE_SRCS})

if(PY_WRAPPER)

    set (PY_WRAPPER_SRCS
        "message_infrastructure/csrc/message_infrastructure_py_wrapper.cc"
        "message_infrastructure/csrc/channel_proxy.cc"
        "message_infrastructure/csrc/port_proxy.cc"
        "message_infrastructure/csrc/abstract_port_implementation.cc"
        )
    target_include_directories(message_infrastructure PUBLIC 
                               ${pybind11_INCLUDE_DIRS}
                               ${PYTHON_INCLUDE_DIRS}
                               ${NUMPY_INCLUDE_DIRS}
                               ${PROJECT_SOURCE_DIR}/message_infrastructure/csrc
                               )

    pybind11_add_module(MessageInfrastructurePywrapper ${PY_WRAPPER_SRCS})
    target_link_libraries(MessageInfrastructurePywrapper PRIVATE message_infrastructure rt)
else()
    target_include_directories(message_infrastructure PUBLIC 
                               ${PROJECT_SOURCE_DIR}/message_infrastructure/csrc
                               )
endif()

# Cpp unit tests using GoogleTest
enable_testing()
add_subdirectory(test)