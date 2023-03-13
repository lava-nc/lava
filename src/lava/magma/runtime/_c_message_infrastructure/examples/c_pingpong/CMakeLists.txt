include_directories(../../csrc)

# find_package(pybind11 REQUIRED)

# set(PROXY_SRCS
#     "../../csrc/channel_proxy.cc"
#     "../../csrc/port_proxy.cc")

# find_package(PythonLibs REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})

# pybind11_add_module(comm SHARED "comm.cc" ${PROXY_SRCS})

# add_library(comm "comm.cc" ${PROXY_SRCS})

# target_link_libraries(comm PRIVATE
#   message_infrastructure
#   ${PYTHON_LIBRARIES}
#   pybind11::headers
# )

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