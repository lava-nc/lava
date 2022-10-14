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
)

target_link_libraries(
  test_messaging_infrastructure
  GTest::gtest_main
  pybind11::embed
  message_infrastructure
  rt
)

include(GoogleTest)
gtest_discover_tests(test_messaging_infrastructure)
