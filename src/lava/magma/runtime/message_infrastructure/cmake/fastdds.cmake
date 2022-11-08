cmake_minimum_required(VERSION 3.14)

include(ExternalProject)
ExternalProject_Add(
  foonathan_memory
  GIT_REPOSITORY https://github.com/eProsima/foonathan_memory_vendor.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/foonathan_memory
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DDS_DESTINATION}
)

ExternalProject_Add(
  fastcdr
  GIT_REPOSITORY https://github.com/eProsima/Fast-CDR.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/fastcdr
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DDS_DESTINATION}
)

ExternalProject_Add(
  fastrtps
  GIT_REPOSITORY https://github.com/eProsima/Fast-DDS.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/fastrtps
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DDS_DESTINATION} -DCMAKE_PREFIX_PATH=${COMMON_DDS_DESTINATION}
)
