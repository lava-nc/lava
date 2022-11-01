# An Example for MessageInfrastructurePywrapper

*Welcome to the messaging refactory project, and this will show how an example run*

```shell
mkdir build
cd build
cmake ..
make
cd ..
source setenv.sh
pytest test
```

```shell
# msg log setting
cmake ../ -DMSG_LOG_LEVEL=debug -DMSG_LOG_FILE_ENABLE=1
# MSG_LOG_LEVEL valid value: all, debug, info, dump, warn, err
# MSG_LOG_FILE_ENABLE valid value:  0, 1
```

## Running Unit Tests (C++ / GoogleTest)
```shell
cd build
cmake -DCMAKE_BUILD_TYPE="Debug" ..
make
ctest

# For printing outputs
ctest --verbose
```

## Build with DDS_CHANNEL
```shell
export LD_LIBRARY_PATH="~/Fast-DDS/install/lib"
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/Fast-DDS/install \
-DDDS_CHANNEL=ON \
-DDDS_BACKEND=fast_dds \
-DDDS_TRANSFER_TYPE=SHARED_MEM ..
make
# DDS_CHANNEL: set "ON" to enable the DDS_CHANNEL support
# CMAKE_INSTALL_PREFIX: DDS install Path
# DDS_BACKEND: dds backend, now support fast_dds only
# DDS_TRANSFER_TYPE: dds transfer type, "SHARED_MEM", "TCP" e.g.
```