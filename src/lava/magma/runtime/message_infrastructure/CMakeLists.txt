cmake_minimum_required(VERSION 3.5)
project(message_passing)

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
    target_link_libraries(MessageInfrastructurePywrapper PRIVATE message_infrastructure)
endif()