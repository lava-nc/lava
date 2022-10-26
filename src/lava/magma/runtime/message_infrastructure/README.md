# Message Infrastructure Library CPP Implementation for LAVA

## Introduction
The message infrastructure library is for LAVA to transfer data. The library provides several method to do commnication for IPC on single host or across multiple hosts.

## Build
Assume you are in `src/lava/magma/runtime/message_infrastructure` folder now.
### 1. Create `build` folder and go into it.
```
$ mkdir build
$ cd build
```
### 2. Build the message infrastructure library according to your requirements using Cmake.

#### (1) If you want to use PythonWrapper of the lib, run the default command:
```
$ cmake ..
```

#### (2) If you do not need to use PythonWrapper of the lib and just use the lib for CPP, run the command:
```
$ cmake .. -DPY_WRAPPER=OFF
```
#### (3) If you want to use GRPC channel, please go into the folder,
`
src/lava/magma/runtime/message_infrastructure/message_infrastructure/csrc/channel/grpc
`

  and run the command:
```
$ mkdir build
$ cd build
$ cmake ..
$ make -j10
```
Then go back to `src/lava/magma/runtime/message_infrastructure/build` and run,
```
$ cmake .. -DGRPC_CHANNEL=ON
```

Note : If your env is using http/https proxy, please unable the proxy to use grpc channel. You could use the commands in your ternimal,
```
$ unset http_proxy
$ unset https_proxy
```
### 3. Compile with makefile
Run the command,
```
$ make -j10
```
## Enable env variables
Run the command,
```
$ cd ..
$ source setenv.sh
```
## Run python test
For example, run the python test for channel usage
```
$ python3 test/test_channel.py
```