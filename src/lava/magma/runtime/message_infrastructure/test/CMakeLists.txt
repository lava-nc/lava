include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(../message_infrastructure/csrc)

add_executable(
  test_messaging_infrastructure
  cpptest/test_multiprocessing.cc
  cpptest/test_channel.cc
  cpptest/test_shm_delivery.cc
  cpptest/test_socket_delivery.cc
  $<$<BOOL:${GRPC_CHANNEL}>:cpptest/test_grpc_delivery.cc>
  $<$<BOOL:${DDS_CHANNEL}>:cpptest/test_ddschannel.cc>)


target_link_libraries(
  test_messaging_infrastructure
  GTest::gtest_main
  message_infrastructure
  rt)

if(FASTDDS_ENABLE)
  add_executable(
    test_dds_to_ros
    cpptest/test_dds_to_ros.cc)

  add_executable(
    test_ros_to_dds
    cpptest/test_ros_to_dds.cc)

  target_link_libraries(
    test_dds_to_ros
    message_infrastructure)

  target_link_libraries(
    test_ros_to_dds
    message_infrastructure)
endif()

include(GoogleTest)
gtest_discover_tests(test_messaging_infrastructure PROPERTIES TIMEOUT 10)