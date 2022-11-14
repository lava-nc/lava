cmake_minimum_required(VERSION 3.14)

include(ExternalProject)

if(NOT DEFINED _Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
  else()
    set(PYTHON_EXECUTABLE "${_Python_EXECUTABLE}")
endif()

add_custom_target(iceoryx ALL
                  DEPENDS ${COMMON_DDS_DESTINATION}/lib/libiceoryx_binding_c.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_hoofs.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_platform.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_posh.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_posh_config.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_posh_gateway.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_posh_roudi.a)
add_custom_command(OUTPUT ${COMMON_DDS_DESTINATION}/lib/libiceoryx_binding_c.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_hoofs.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_platform.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_posh.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_posh_config.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_posh_gateway.a
                          ${COMMON_DDS_DESTINATION}/lib/libiceoryx_posh_roudi.a
                   COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_iceoryx.py" ${COMMON_DDS_DESTINATION}  ${CMAKE_BINARY_DIR})

ExternalProject_Add(
  cyclonedds
  GIT_REPOSITORY https://github.com/eclipse-cyclonedds/cyclonedds.git
  GIT_TAG 0.10.2
  SOURCE_DIR ${CMAKE_BINARY_DIR}/cyclonedds
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DDS_DESTINATION} -DENABLE_SHM=ON
)

add_dependencies(cyclonedds iceoryx)

ExternalProject_Add(
  cyclonedds-cxx
  GIT_REPOSITORY https://github.com/eclipse-cyclonedds/cyclonedds-cxx.git
  GIT_TAG 0.10.2
  SOURCE_DIR ${CMAKE_BINARY_DIR}/cyclonedds-cxx
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DDS_DESTINATION}
)

add_dependencies(cyclonedds-cxx cyclonedds)
