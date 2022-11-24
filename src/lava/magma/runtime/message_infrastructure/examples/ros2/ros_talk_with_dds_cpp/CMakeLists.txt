cmake_minimum_required(VERSION 3.5)
project(ros_talk_with_dds_cpp)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ddsmetadata REQUIRED)  

include_directories(include)

add_executable(ros_pub src/publisher.cpp)
ament_target_dependencies(ros_pub rclcpp ddsmetadata)

add_executable(ros_sub src/subscriber.cpp)
ament_target_dependencies(ros_sub rclcpp ddsmetadata)

install(TARGETS
  ros_pub
  ros_sub
  DESTINATION lib/${PROJECT_NAME})

ament_package()
