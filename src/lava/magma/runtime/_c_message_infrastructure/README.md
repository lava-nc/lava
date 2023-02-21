# Message Infrastructure Library CPP Implementation for LAVA
version: v0.2.1
## Introduction
The message infrastructure library is for LAVA to transfer data. The library provides several method to do communication for IPC on single host or across multiple hosts.

## Build
Assume you are in `<lava>/` folder now.
### 1. Set cmake args by env variables to build the message infrastructure library according to your requirements.
```bash
$ export CMAKE_ARGS="..."
```
#### (1) If you want to use PythonWrapper of the lib, this step could be just ignored as this is the default setting.
#### (2) If you do not need to use PythonWrapper of the lib and just use the lib for CPP, run the command:
```bash
$ export CMAKE_ARGS="-DPY_WRAPPER=OFF"
```
#### (3) If you want to use GRPC channel, run the command:

```bash
$ export CMAKE_ARGS="-DGRPC_CHANNEL=ON"
```

Note :
-  If your env is using http/https proxy, please unable the proxy to use grpc channel.<br>
You could use the commands in your ternimal,
  ```bash
  $ unset http_proxy
  $ unset https_proxy
  ```
-  When you use grpc channel at main and sub processes together, pls refer to [this link](https://github.com/grpc/grpc/blob/master/doc/fork_support.md) to set env.
-  There are conflict of `LOCKABLE` definition at CycloneDDS and gRPC, so reject enabling GRPC_CHANNEL and CycloneDDS_ENABLE together.

#### (4) If you want to enable DDS channel, run the command:
```bash
$ export CMAKE_ARGS="-DDDS_CHANNEL=ON -D<DDS_BACKEND>_ENABLE=ON"
# [DDS_BACKEND: FASTDDS, CycloneDDS ..., only support FASTDDS now]
# Before build FastDDS, need to install dependences by below command.
# sudo apt-get install libasio-dev libtinyxml2-dev
``` 

#### (5) Build with cpp unit tests

```bash
$ export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug"
```

### 2. Compile library code
-  Run the command to build message infrastructure library.
```bash
$ python3 prebuild.py
```
-  If you have select to use PythonWrapper, GRPC channel, DDS channel or CPP unit tests, the source code will be compiled together with the message infrastructure library code.
### 3. Add PYTHONPATH
-  Add PYTHONPATH into terminal environment.
```bash
$  export PYTHONPATH=src/:$PYTHONPATH
```
## Run Python test
-  For example, run the python test for channel usage
  ```bash
  $ python3 tests/lava/magma/runtime/message_infrastructure/test_channel.py
  ```
-  Run all tests
  ```bash
  # when enable grpc channel, need to add following env:
  # export GRPC_ENABLE_FORK_SUPPORT=true
  # export GRPC_POLL_STRATEGY=poll
  $ pytest tests/lava/magma/runtime/message_infrastructure/
  ```

## Run CPP test
-  Run all the CPP test for msg lib
  ```bash
  $ build/test/test_messaging_infrastructure
  ```

## Install by poetry
Also users could choose to use poetry to enbale the whole environment.
```bash
$ export CMAKE_ARGS="..."
$ poetry install
$ source .venc/bin/activate
```
