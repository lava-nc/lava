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
    "message_infrastructure/csrc/core/message_infrastructure_logging.cc"
    "message_infrastructure/csrc/actor/posix_actor.cc"
    "message_infrastructure/csrc/channel/shmem/shm.cc"
    "message_infrastructure/csrc/channel/shmem/shmem_channel.cc"
    "message_infrastructure/csrc/channel/shmem/shmem_port.cc"
    "message_infrastructure/csrc/channel/socket/socket.cc"
    "message_infrastructure/csrc/channel/socket/socket_channel.cc"
    "message_infrastructure/csrc/channel/socket/socket_port.cc"
    )

add_library(message_infrastructure SHARED ${MESSAGE_INFRASTRUCTURE_SRCS})

set(MSG_LOG_LEVEL 16 CACHE STRING "Default vaule: 16, error log only.")
set(MSG_LOG_PRINT_MODE 1 CACHE STRING "Default value: 1, print onto console only.")

if ((${MSG_LOG_LEVEL} LESS 32) AND (${MSG_LOG_LEVEL} GREATER_EQUAL 0))
    target_compile_definitions(message_infrastructure PRIVATE
                               MSG_LOG_LEVEL=${MSG_LOG_LEVEL}
                               )
else()
    target_compile_definitions(message_infrastructure PRIVATE
                               $<$<STREQUAL:${MSG_LOG_LEVEL},all>:MSG_LOG_LEVEL_ALL>
                               $<$<STREQUAL:${MSG_LOG_LEVEL},err_warn>:MSG_LOG_LEVEL_WARN_ERR>
                               )
endif()

if ((${MSG_LOG_PRINT_MODE} LESS 4) AND (${MSG_LOG_PRINT_MODE} GREATER_EQUAL 0))
    target_compile_definitions(message_infrastructure PRIVATE
                               MSG_LOG_PRINT_MODE=${MSG_LOG_PRINT_MODE}
                               )
else()
target_compile_definitions(message_infrastructure PRIVATE
                           $<$<STREQUAL:${MSG_LOG_PRINT_MODE},all>:MSG_LOG_PRINT_MODE_ALL>
                           $<$<STREQUAL:${MSG_LOG_PRINT_MODE},file>:MSG_LOG_PRINT_MODE_FILE>
                           $<$<STREQUAL:${MSG_LOG_PRINT_MODE},shell>:MSG_LOG_PRINT_MODE_SHEL>
                           $<$<STREQUAL:${MSG_LOG_PRINT_MODE},null>:MSG_LOG_PRINT_MODE_NULL>
                           )
endif()

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

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("debug mode and enable cpp unit test")
    enable_testing()
    add_subdirectory(test)
else()
    message("not debug mode and disable cpp unit test")
endif()
