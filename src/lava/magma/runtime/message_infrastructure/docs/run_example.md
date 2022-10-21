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
