# Message Infrastructure Library CPP Implementation for LAVA

## Introduction
The message infrastructure library is for LAVA to transfer data. The library provides several method to do commnication for IPC on single host or across multiple hosts.

## Build
Assume you are in `src/lava/magma/runtime/message_infrastructure` folder now.
### 1. Create `build` folder and go into it.
```bash
$ mkdir build
$ cd build
```
### 2. Build the message infrastructure library according to your requirements using Cmake.

#### (1) If you want to use PythonWrapper of the lib, run the default command:
```bash
# set numpy dependence by -DNUMPY_INCLUDE_DIRS=<numpy_include_dir>
$ cmake ..
```

#### (2) If you do not need to use PythonWrapper of the lib and just use the lib for CPP, run the command:
```bash
$ cmake .. -DPY_WRAPPER=OFF
```
#### (3) If you want to use GRPC channel, run the command:

```bash
$ cmake .. -DGRPC_CHANNEL=ON
```

Note : If your env is using http/https proxy, please unable the proxy to use grpc channel.<br>
You could use the commands in your ternimal,
```bash
$ unset http_proxy
$ unset https_proxy
```
#### (4) If you want to enable DDS channel, run the command:
```bash
$cmake .. -DDDS_CHANNEL=ON -D<DDS_BACKEND>_ENABLE=ON
# [DDS_BACKEND: FASTDDS, CycloneDDS ..., only support FASTDDS now]
``` 
### 3. Compile with makefile
Run the command,
```bash
$ make -j<parallel_num>
```
## Enable env variables
Run the command,
```bash
$ cd ..
$ source setenv.sh
```
## Run python test
- For example, run the python test for channel usage
  ```bash
  $ python3 test/test_channel.py
  ```
- Run all tests
  ```bash
  $ pytest test/
  ```

## Install by poetry

```bash
# Enable grpc channel
# export CMAKE_ARGS="-DGRPC_CHANNEL=ON"
$ poetry install
```