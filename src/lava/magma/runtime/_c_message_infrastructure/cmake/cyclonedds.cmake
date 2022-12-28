cmake_minimum_required(VERSION 3.14)

include(ExternalProject)

ExternalProject_Add(
  cyclonedds
  GIT_REPOSITORY https://github.com/eclipse-cyclonedds/cyclonedds.git
  GIT_TAG 0.10.2
  SOURCE_DIR ${CMAKE_BINARY_DIR}/cyclonedds
  # CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DDS_DESTINATION}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DDS_DESTINATION} -DBUILD_SHARED_LIBS=OFF -DENABLE_SSL=OFF -DENABLE_SECURITY=OFF -D-DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_C_FLAGS="-fPIC"
)

ExternalProject_Add(
  cyclonedds-cxx
  GIT_REPOSITORY https://github.com/eclipse-cyclonedds/cyclonedds-cxx.git
  GIT_TAG 0.10.2
  SOURCE_DIR ${CMAKE_BINARY_DIR}/cyclonedds-cxx
  # CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DDS_DESTINATION}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${COMMON_DDS_DESTINATION} -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_C_FLAGS="-fPIC"
)

add_dependencies(cyclonedds-cxx cyclonedds)
