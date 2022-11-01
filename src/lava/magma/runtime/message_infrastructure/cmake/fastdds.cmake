cmake_minimum_required(VERSION 3.14)

set(COMMON_DESTINATION "${CMAKE_BINARY_DIR}/install")

include(ExternalProject)
ExternalProject_Add(
  foonathan_memory
  GIT_REPOSITORY https://github.com/eProsima/foonathan_memory_vendor.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/foonathan_memory
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DESTINATION} -DBUILD_SHARED_LIBS=ON
)
add_dependencies(dds_channel foonathan_memory)

target_link_libraries(dds_channel INTERFACE
  # consider the lib name version number
  ${COMMON_DESTINATION}/lib/libfoonathan_memory-0.7.1.so
)

ExternalProject_Add(
  fastcdr
  GIT_REPOSITORY https://github.com/eProsima/Fast-CDR.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/fastcdr
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DESTINATION}
)
target_include_directories(dds_channel INTERFACE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/fastcdr>
    $<INSTALL_INTERFACE:include>
)

add_dependencies(dds_channel fastcdr)
target_link_libraries(dds_channel INTERFACE
  ${COMMON_DESTINATION}/lib/libfastcdr.so
)

ExternalProject_Add(
  fastrtps
  GIT_REPOSITORY https://github.com/eProsima/Fast-DDS.git
  SOURCE_DIR ${CMAKE_BINARY_DIR}/fastrtps
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DESTINATION} -DCMAKE_PREFIX_PATH=${COMMON_DESTINATION}
)
add_dependencies(dds_channel fastrtps)
target_include_directories(dds_channel INTERFACE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/fastrtps/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(dds_channel INTERFACE
  ${COMMON_DESTINATION}/lib/libfastrtps.so
)

list(APPEND CMAKE_PREFIX_PATH ${COMMON_DESTINATION})
