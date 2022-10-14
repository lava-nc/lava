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

## Running Unit Tests (C++ / GoogleTest)
```shell
cd build
cmake -DCMAKE_BUILD_TYPE="Debug" ..
make
ctest

# For printing outputs
ctest --verbose
```
