include_directories(../../csrc)

add_executable(cprocess
  "cprocess.cc"
)

target_link_libraries(cprocess PRIVATE
  message_infrastructure
)


set_target_properties(cprocess
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)