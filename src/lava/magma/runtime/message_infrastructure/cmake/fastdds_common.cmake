cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_definitions(-D_WIN32_WINNT=0x600)
endif()

find_package(Threads REQUIRED)
include(FetchContent)

if(FastDDS_FETCHCONTENT)
  message(STATUS "Using DDS via add_subdirectory (FetchContent).")
  FetchContent_Declare(
    foonathan_memory
    GIT_REPOSITORY https://github.com/eProsima/foonathan_memory_vendor.git

    )

  FetchContent_Declare(
    fastcdr
    GIT_REPOSITORY https://github.com/eProsima/Fast-CDR.git
    )

  FetchContent_Declare(
    fastrtps
    GIT_REPOSITORY https://github.com/eProsima/Fast-DDS.git
    )
  FetchContent_MakeAvailable(foonathan_memory fastcdr fastrtps)

else()
  find_package(foonathan_memory REQUIRED)
  find_package(fastcdr REQUIRED)
  find_package(fastrtps REQUIRED)
endif()