include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(../csrc)

add_executable(
  test_messaging_infrastructure
  test_multiprocessing.cc
  test_channel.cc
  test_shm_delivery.cc
  test_socket_delivery.cc
  $<$<BOOL:${GRPC_CHANNEL}>:test_grpc_delivery.cc>
  $<$<BOOL:${DDS_CHANNEL}>:test_ddschannel.cc>)


target_link_libraries(
  test_messaging_infrastructure
  GTest::gtest_main
  message_infrastructure
  rt)

if(FASTDDS_ENABLE)
  add_executable(
    test_fastdds_to_ros
    test_fastdds_to_ros.cc)

  add_executable(
    test_fastdds_from_ros
    test_fastdds_from_ros.cc)

  target_link_libraries(
    test_fastdds_to_ros
    message_infrastructure)

  target_link_libraries(
    test_fastdds_from_ros
    message_infrastructure)

elseif(CycloneDDS_ENABLE)
  add_executable(
    test_cyclonedds_to_ros
    test_cyclonedds_to_ros.cc)

  add_executable(
    test_cyclonedds_from_ros
    test_cyclonedds_from_ros.cc)

  target_link_libraries(
    test_cyclonedds_to_ros
    message_infrastructure)

  target_link_libraries(
    test_cyclonedds_from_ros
    message_infrastructure)
endif()

include(GoogleTest)
gtest_discover_tests(test_messaging_infrastructure PROPERTIES TIMEOUT 10)